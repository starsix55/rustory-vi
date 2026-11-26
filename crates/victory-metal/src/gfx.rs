//! Descriptor heaps, textures, samplers, render pipelines and passes on Metal, plus the one encoder
//! loop (`submit_passes`) that compute `submit` also goes through.
//!
//! Descriptor encoding is `metal_irconverter_runtime.h` re-implemented from its documented layout
//! (`IRDescriptorTableSetBuffer` / `SetTexture` / `SetSampler`, 24-byte entries), so no Apple header
//! is needed at build time.

use crate::backend::{ComputeBackend, Heaps, RootArgs};
use crate::metal::{Buffer, Metal, Queue};
use crate::resources::*;
use objc2::Message;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::*;
use std::cell::{Cell, RefCell};
use std::ptr::NonNull;

type Res = Retained<ProtocolObject<dyn MTLResource>>;

/// A fence chaining the encoders of one command buffer, and whether one has run: what a pass waits
/// for and updates when the queue's buffers are made resident by a set instead of named.
struct Order {
    fence: Option<Retained<ProtocolObject<dyn MTLFence>>>,
    started: Cell<bool>,
}

type Ordering<'a> = Option<(&'a ProtocolObject<dyn MTLFence>, &'a Cell<bool>)>;

/// `IRDescriptorTableEntry`: `{ gpuVA, textureViewID, metadata }`.
const ENTRY: usize = 24;

/// `kIRDescriptorHeapBindPoint`, `kIRSamplerHeapBindPoint`, `kIRArgumentBufferBindPoint`,
/// `kIRArgumentBufferDrawArgumentsBindPoint`, `kIRArgumentBufferUniformsBindPoint`,
/// `kIRVertexBufferBindPoint`, `kIRStageInAttributeStartIndex` (`metal_irconverter_runtime.h`).
const HEAP_SLOT: [usize; 2] = [0, 1];
const ARGUMENT_BUFFER: usize = 2;
const DRAW_ARGUMENTS: usize = 4;
const UNIFORMS: usize = 5;
const VERTEX_BUFFERS: usize = 6;
const STAGE_IN_ATTRIBUTES: usize = 11;

pub struct Heap {
    buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    kind: HeapKind,
    count: usize,
    /// What each entry names, kept alive here and made resident with the heap.
    refs: RefCell<std::collections::HashMap<usize, Res>>,
    /// Residency set holding the heap buffer and everything its entries name.
    set: Option<Retained<ProtocolObject<dyn MTLResidencySet>>>,
    dirty: Cell<bool>,
}

pub struct Texture {
    tex: Retained<ProtocolObject<dyn MTLTexture>>,
}

impl Texture {
    /// The texture as the bytes it holds when it is sRGB (a view in the plain format), else itself.
    pub(crate) fn view_without_srgb(&self, _m: &Metal) -> Retained<ProtocolObject<dyn MTLTexture>> {
        let plain = match self.tex.pixelFormat() {
            MTLPixelFormat::RGBA8Unorm_sRGB => MTLPixelFormat::RGBA8Unorm,
            MTLPixelFormat::BGRA8Unorm_sRGB => MTLPixelFormat::BGRA8Unorm,
            _ => return self.tex.clone(),
        };
        self.tex
            .newTextureViewWithPixelFormat(plain)
            .unwrap_or_else(|| self.tex.clone())
    }
}

pub struct Sampler(Retained<ProtocolObject<dyn MTLSamplerState>>);

pub struct RenderPipeline {
    pso: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    depth: Option<Retained<ProtocolObject<dyn MTLDepthStencilState>>>,
    cull: MTLCullMode,
    raster: Raster,
    /// The pipeline's target has a stencil plane (the pass attaches it too).
    stencil: bool,
}

fn pixel_format(f: Format) -> MTLPixelFormat {
    match f {
        Format::Rgba8Unorm => MTLPixelFormat::RGBA8Unorm,
        Format::Rgba8UnormSrgb => MTLPixelFormat::RGBA8Unorm_sRGB,
        Format::Bgra8Unorm => MTLPixelFormat::BGRA8Unorm,
        Format::Rgba16Float => MTLPixelFormat::RGBA16Float,
        Format::R32Float => MTLPixelFormat::R32Float,
        Format::D32Float => MTLPixelFormat::Depth32Float,
        Format::D32FloatS8 => MTLPixelFormat::Depth32Float_Stencil8,
        Format::Bc1 => MTLPixelFormat::BC1_RGBA,
        Format::Bc2 => MTLPixelFormat::BC2_RGBA,
        Format::Bc3 => MTLPixelFormat::BC3_RGBA,
        Format::Bc4 => MTLPixelFormat::BC4_RUnorm,
        Format::Bc5 => MTLPixelFormat::BC5_RGUnorm,
        Format::Bc6h => MTLPixelFormat::BC6H_RGBFloat,
        Format::Bc7 => MTLPixelFormat::BC7_RGBAUnorm,
        Format::R8Unorm => MTLPixelFormat::R8Unorm,
        Format::A8Unorm => MTLPixelFormat::A8Unorm,
        Format::R16Unorm => MTLPixelFormat::R16Unorm,
        Format::Rgba32Float => MTLPixelFormat::RGBA32Float,
        Format::Rg32Float => MTLPixelFormat::RG32Float,
        Format::Rgb10a2Unorm => MTLPixelFormat::RGB10A2Unorm,
        Format::Rg11b10Float => MTLPixelFormat::RG11B10Float,
        Format::Rgba8Uint => MTLPixelFormat::RGBA8Uint,
        Format::Rg16Float => MTLPixelFormat::RG16Float,
        Format::Rg16Unorm => MTLPixelFormat::RG16Unorm,
        Format::R32Uint => MTLPixelFormat::R32Uint,
        Format::R8Uint => MTLPixelFormat::R8Uint,
        Format::Rgb9e5Float => MTLPixelFormat::RGB9E5Float,
        Format::Bc7Srgb => MTLPixelFormat::BC7_RGBAUnorm_sRGB,
        Format::Rgba32Uint => MTLPixelFormat::RGBA32Uint,
        Format::Rgba16Unorm => MTLPixelFormat::RGBA16Unorm,
        Format::Rgba16Snorm => MTLPixelFormat::RGBA16Snorm,
        Format::Rg32Uint => MTLPixelFormat::RG32Uint,
        Format::Rg16Uint => MTLPixelFormat::RG16Uint,
        Format::Rg8Unorm => MTLPixelFormat::RG8Unorm,
        Format::Rg8Uint => MTLPixelFormat::RG8Uint,
        Format::R16Float => MTLPixelFormat::R16Float,
        Format::R16Uint => MTLPixelFormat::R16Uint,
        Format::D16Unorm => MTLPixelFormat::Depth16Unorm,
        Format::Bc1Srgb => MTLPixelFormat::BC1_RGBA_sRGB,
        Format::Bc3Srgb => MTLPixelFormat::BC3_RGBA_sRGB,
        Format::Bgr5a1Unorm => MTLPixelFormat::BGR5A1Unorm,
        Format::Abgr4Unorm => MTLPixelFormat::ABGR4Unorm,
    }
}

fn tex_type(d: TexDim) -> MTLTextureType {
    match d {
        TexDim::D2 => MTLTextureType::Type2D,
        TexDim::D2Array => MTLTextureType::Type2DArray,
        TexDim::D3 => MTLTextureType::Type3D,
    }
}

fn compare(c: Compare) -> MTLCompareFunction {
    match c {
        Compare::Never => MTLCompareFunction::Never,
        Compare::Less => MTLCompareFunction::Less,
        Compare::Equal => MTLCompareFunction::Equal,
        Compare::LessEqual => MTLCompareFunction::LessEqual,
        Compare::Greater => MTLCompareFunction::Greater,
        Compare::NotEqual => MTLCompareFunction::NotEqual,
        Compare::GreaterEqual => MTLCompareFunction::GreaterEqual,
        Compare::Always => MTLCompareFunction::Always,
    }
}

fn blend_factor(f: BlendFactor) -> MTLBlendFactor {
    match f {
        BlendFactor::Zero => MTLBlendFactor::Zero,
        BlendFactor::One => MTLBlendFactor::One,
        BlendFactor::SrcColor => MTLBlendFactor::SourceColor,
        BlendFactor::InvSrcColor => MTLBlendFactor::OneMinusSourceColor,
        BlendFactor::SrcAlpha => MTLBlendFactor::SourceAlpha,
        BlendFactor::InvSrcAlpha => MTLBlendFactor::OneMinusSourceAlpha,
        BlendFactor::DestAlpha => MTLBlendFactor::DestinationAlpha,
        BlendFactor::InvDestAlpha => MTLBlendFactor::OneMinusDestinationAlpha,
        BlendFactor::DestColor => MTLBlendFactor::DestinationColor,
        BlendFactor::InvDestColor => MTLBlendFactor::OneMinusDestinationColor,
        BlendFactor::SrcAlphaSat => MTLBlendFactor::SourceAlphaSaturated,
        BlendFactor::Constant => MTLBlendFactor::BlendColor,
        BlendFactor::InvConstant => MTLBlendFactor::OneMinusBlendColor,
        BlendFactor::Src1Color => MTLBlendFactor::Source1Color,
        BlendFactor::InvSrc1Color => MTLBlendFactor::OneMinusSource1Color,
        BlendFactor::Src1Alpha => MTLBlendFactor::Source1Alpha,
        BlendFactor::InvSrc1Alpha => MTLBlendFactor::OneMinusSource1Alpha,
    }
}

fn blend_op(o: BlendOp) -> MTLBlendOperation {
    match o {
        BlendOp::Add => MTLBlendOperation::Add,
        BlendOp::Subtract => MTLBlendOperation::Subtract,
        BlendOp::ReverseSubtract => MTLBlendOperation::ReverseSubtract,
        BlendOp::Min => MTLBlendOperation::Min,
        BlendOp::Max => MTLBlendOperation::Max,
    }
}

fn stencil_op(o: StencilOp) -> MTLStencilOperation {
    match o {
        StencilOp::Keep => MTLStencilOperation::Keep,
        StencilOp::Zero => MTLStencilOperation::Zero,
        StencilOp::Replace => MTLStencilOperation::Replace,
        StencilOp::IncrementClamp => MTLStencilOperation::IncrementClamp,
        StencilOp::DecrementClamp => MTLStencilOperation::DecrementClamp,
        StencilOp::Invert => MTLStencilOperation::Invert,
        StencilOp::IncrementWrap => MTLStencilOperation::IncrementWrap,
        StencilOp::DecrementWrap => MTLStencilOperation::DecrementWrap,
    }
}

fn stencil_face(f: &StencilFace, s: &StencilState) -> Retained<MTLStencilDescriptor> {
    let d = MTLStencilDescriptor::new();
    d.setStencilFailureOperation(stencil_op(f.fail));
    d.setDepthFailureOperation(stencil_op(f.depth_fail));
    d.setDepthStencilPassOperation(stencil_op(f.pass));
    d.setStencilCompareFunction(compare(f.compare));
    d.setReadMask(u32::from(s.read_mask));
    d.setWriteMask(u32::from(s.write_mask));
    d
}

fn address(a: Address) -> MTLSamplerAddressMode {
    match a {
        Address::Wrap => MTLSamplerAddressMode::Repeat,
        Address::Mirror => MTLSamplerAddressMode::MirrorRepeat,
        Address::Clamp => MTLSamplerAddressMode::ClampToEdge,
    }
}

fn filter(f: Filter) -> MTLSamplerMinMagFilter {
    match f {
        Filter::Nearest => MTLSamplerMinMagFilter::Nearest,
        Filter::Linear => MTLSamplerMinMagFilter::Linear,
    }
}

fn vertex_format(f: VertexFormat) -> MTLVertexFormat {
    match f {
        VertexFormat::Float => MTLVertexFormat::Float,
        VertexFormat::Float2 => MTLVertexFormat::Float2,
        VertexFormat::Float3 => MTLVertexFormat::Float3,
        VertexFormat::Float4 => MTLVertexFormat::Float4,
        VertexFormat::Uint => MTLVertexFormat::UInt,
        VertexFormat::Uint2 => MTLVertexFormat::UInt2,
        VertexFormat::Uint3 => MTLVertexFormat::UInt3,
        VertexFormat::Uint4 => MTLVertexFormat::UInt4,
        VertexFormat::Sint => MTLVertexFormat::Int,
        VertexFormat::Sint2 => MTLVertexFormat::Int2,
        VertexFormat::Sint3 => MTLVertexFormat::Int3,
        VertexFormat::Sint4 => MTLVertexFormat::Int4,
        VertexFormat::Half => MTLVertexFormat::Half,
        VertexFormat::Half2 => MTLVertexFormat::Half2,
        VertexFormat::Half4 => MTLVertexFormat::Half4,
        VertexFormat::Unorm16 => MTLVertexFormat::UShortNormalized,
        VertexFormat::Unorm16x2 => MTLVertexFormat::UShort2Normalized,
        VertexFormat::Unorm16x4 => MTLVertexFormat::UShort4Normalized,
        VertexFormat::Snorm16 => MTLVertexFormat::ShortNormalized,
        VertexFormat::Snorm16x2 => MTLVertexFormat::Short2Normalized,
        VertexFormat::Snorm16x4 => MTLVertexFormat::Short4Normalized,
        VertexFormat::Uint16 => MTLVertexFormat::UShort,
        VertexFormat::Uint16x2 => MTLVertexFormat::UShort2,
        VertexFormat::Uint16x4 => MTLVertexFormat::UShort4,
        VertexFormat::Sint16 => MTLVertexFormat::Short,
        VertexFormat::Sint16x2 => MTLVertexFormat::Short2,
        VertexFormat::Sint16x4 => MTLVertexFormat::Short4,
        VertexFormat::Unorm8 => MTLVertexFormat::UCharNormalized,
        VertexFormat::Unorm8x2 => MTLVertexFormat::UChar2Normalized,
        VertexFormat::Unorm8x4 => MTLVertexFormat::UChar4Normalized,
        VertexFormat::Snorm8 => MTLVertexFormat::CharNormalized,
        VertexFormat::Snorm8x2 => MTLVertexFormat::Char2Normalized,
        VertexFormat::Snorm8x4 => MTLVertexFormat::Char4Normalized,
        VertexFormat::Uint8 => MTLVertexFormat::UChar,
        VertexFormat::Uint8x2 => MTLVertexFormat::UChar2,
        VertexFormat::Uint8x4 => MTLVertexFormat::UChar4,
        VertexFormat::Sint8 => MTLVertexFormat::Char,
        VertexFormat::Sint8x2 => MTLVertexFormat::Char2,
        VertexFormat::Sint8x4 => MTLVertexFormat::Char4,
        VertexFormat::Unorm1010102 => MTLVertexFormat::UInt1010102Normalized,
        VertexFormat::Snorm1010102 => MTLVertexFormat::Int1010102Normalized,
        VertexFormat::FloatRG11B10 => MTLVertexFormat::FloatRG11B10,
    }
}

/// `MTLResourceID` is a `#[repr(transparent)]` u64 whose field is crate-private in `objc2-metal`.
fn resource_id(id: MTLResourceID) -> u64 {
    // SAFETY: MTLResourceID is one u64 (`Encode`d as such).
    unsafe { std::mem::transmute::<MTLResourceID, u64>(id) }
}

fn size(a: [usize; 3]) -> MTLSize {
    MTLSize {
        width: a[0],
        height: a[1],
        depth: a[2],
    }
}

impl Metal {
    fn heap_entry(&self, heap: &Heap, index: usize, words: [u64; 3], keep: Option<Res>) {
        assert!(
            index < heap.count,
            "descriptor index {index} out of {}",
            heap.count
        );
        // SAFETY: `index` is inside the heap buffer, which is shared memory the GPU is not
        // reading while the title writes descriptors (D3D12 has the same rule).
        unsafe {
            let p = heap
                .buf
                .contents()
                .as_ptr()
                .cast::<u64>()
                .add(index * ENTRY / 8);
            words
                .iter()
                .enumerate()
                .for_each(|(i, w)| p.add(i).write_volatile(*w));
        }
        if let Some(r) = keep {
            if let Some(set) = &heap.set {
                set.addAllocation(ProtocolObject::from_ref(&*r));
                heap.dirty.set(true);
            }
            heap.refs.borrow_mut().insert(index, r);
        }
    }

    /// Residency for one command buffer: the heap's set, committed if entries changed.
    fn use_heap_cb(&self, cb: &ProtocolObject<dyn MTLCommandBuffer>, heap: &Heap) {
        if let Some(set) = &heap.set {
            if heap.dirty.replace(false) {
                set.commit();
            }
            cb.useResidencySet(set);
        }
    }

    /// Residency without sets: every entry's resource is named to the encoder.
    fn use_heap_encoder(
        &self,
        heap: &Heap,
        mut use_res: impl FnMut(&ProtocolObject<dyn MTLResource>),
    ) {
        if heap.set.is_some() {
            return;
        }
        use_res(ProtocolObject::from_ref(&*heap.buf));
        heap.refs.borrow().values().for_each(|r| use_res(r));
    }

    fn stage_function(
        &self,
        dxil: &[u8],
        layout: Option<&victory_backend::backend::RootLayout>,
    ) -> Result<Retained<ProtocolObject<dyn MTLFunction>>, String> {
        let (_, lib) = self.msc.convert_with(dxil, None, layout).map_err(|e| e.to_string())?;
        let l = self.load_library(&lib)?;
        let name = l
            .functionNames()
            .iter()
            .next()
            .ok_or("converted library is empty")?;
        l.newFunctionWithName(&name)
            .ok_or_else(|| "no function".into())
    }

    fn bind_root_compute(
        &self,
        e: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        root: &RootArgs,
        heaps: &Heaps<'_, Metal>,
    ) {
        if !root.0.is_empty() {
            // SAFETY: setBytes copies `root` (at most 4 KiB) before returning.
            unsafe {
                e.setBytes_length_atIndex(
                    NonNull::new(root.0.as_ptr().cast_mut().cast()).unwrap(),
                    root.0.len(),
                    ARGUMENT_BUFFER,
                )
            };
        }
        for (h, slot) in [
            (heaps.resource, HEAP_SLOT[0]),
            (heaps.sampler, HEAP_SLOT[1]),
        ] {
            if let Some(h) = h {
                // SAFETY: a live buffer at an in-range slot.
                unsafe { e.setBuffer_offset_atIndex(Some(&h.buf), 0, slot) };
                self.use_heap_encoder(h, |r| {
                    e.useResource_usage(r, MTLResourceUsage::Read | MTLResourceUsage::Write)
                });
            }
        }
    }

    fn bind_root_render(
        &self,
        e: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        root: &RootArgs,
        fragment: Option<&RootArgs>,
        heaps: &Heaps<'_, Metal>,
    ) {
        let fragment = fragment.unwrap_or(root);
        // SAFETY: setBytes copies the bytes before returning.
        unsafe {
            if !root.0.is_empty() {
                let p = NonNull::new(root.0.as_ptr().cast_mut().cast()).unwrap();
                e.setVertexBytes_length_atIndex(p, root.0.len(), ARGUMENT_BUFFER);
            }
            if !fragment.0.is_empty() {
                let p = NonNull::new(fragment.0.as_ptr().cast_mut().cast()).unwrap();
                e.setFragmentBytes_length_atIndex(p, fragment.0.len(), ARGUMENT_BUFFER);
            }
        }
        for (h, slot) in [
            (heaps.resource, HEAP_SLOT[0]),
            (heaps.sampler, HEAP_SLOT[1]),
        ] {
            if let Some(h) = h {
                // SAFETY: a live buffer at an in-range slot.
                unsafe {
                    e.setVertexBuffer_offset_atIndex(Some(&h.buf), 0, slot);
                    e.setFragmentBuffer_offset_atIndex(Some(&h.buf), 0, slot);
                }
                self.use_heap_encoder(h, |r| {
                    e.useResource_usage_stages(
                        r,
                        MTLResourceUsage::Read | MTLResourceUsage::Write,
                        MTLRenderStages::Vertex | MTLRenderStages::Fragment,
                    )
                });
            }
        }
    }
}

impl ResourceBackend for Metal {
    type Texture = Texture;
    type Sampler = Sampler;

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            placement_heaps: false,
            sparse_textures: false,
            memoryless_attachments: false,
            texture_views: true,
            linear_texture_views: true,
        }
    }

    fn allocated_size(&self, t: &Texture) -> usize {
        // `MTLTexture` inherits `allocatedSize` from `MTLResource`, but the
        // generated bindings do not expose the supertrait on the protocol
        // object, so ask the object directly.
        unsafe { objc2::msg_send![&*t.tex, allocatedSize] }
    }

    fn create_heap(&self, kind: HeapKind, count: usize) -> Result<Heap, String> {
        let buf = self
            .dev
            .newBufferWithLength_options(count * ENTRY, MTLResourceOptions::StorageModeShared)
            .ok_or("descriptor heap allocation failed")?;
        let set = if self.residency_sets {
            let d = MTLResidencySetDescriptor::new();
            let s = self
                .dev
                .newResidencySetWithDescriptor_error(&d)
                .map_err(|e| e.localizedDescription().to_string())?;
            s.addAllocation(ProtocolObject::from_ref(&*buf));
            s.commit();
            Some(s)
        } else {
            None
        };
        Ok(Heap {
            buf,
            kind,
            count,
            refs: RefCell::default(),
            set,
            dirty: Cell::new(false),
        })
    }

    fn heap_address(&self, heap: &Heap, index: usize) -> u64 {
        heap.buf.gpuAddress() + (index * ENTRY) as u64
    }

    fn write_buffer(
        &self,
        heap: &Heap,
        index: usize,
        buffer: &Buffer,
        title: *const u8,
        size: u32,
    ) {
        debug_assert_eq!(heap.kind, HeapKind::Resource);
        let gpu = self.gpu_address(buffer, title);
        let keep: Res = ProtocolObject::from_ref(&*buffer.buf).retain();
        // IRDescriptorTableSetBuffer(entry, gpu, metadata = size & 0xffffffff).
        self.heap_entry(heap, index, [gpu, 0, size as u64], Some(keep));
    }

    fn write_texture(&self, heap: &Heap, index: usize, t: &Texture, min_lod: f32) {
        debug_assert_eq!(heap.kind, HeapKind::Resource);
        let id = resource_id(t.tex.gpuResourceID());
        let keep: Res = ProtocolObject::from_ref(&*t.tex).retain();
        // IRDescriptorTableSetTexture(entry, tex, minLODClamp, metadata = 0).
        self.heap_entry(heap, index, [0, id, min_lod.to_bits() as u64], Some(keep));
    }

    fn write_sampler(&self, heap: &Heap, index: usize, s: &Sampler, lod_bias: f32) {
        debug_assert_eq!(heap.kind, HeapKind::Sampler);
        let id = resource_id(s.0.gpuResourceID());
        // IRDescriptorTableSetSampler(entry, sampler, lodBias): samplers need no residency.
        self.heap_entry(heap, index, [id, 0, lod_bias.to_bits() as u64], None);
    }

    fn create_texture(&self, d: &TextureDesc) -> Result<Texture, String> {
        let td = MTLTextureDescriptor::new();
        td.setTextureType(tex_type(d.dim));
        td.setPixelFormat(pixel_format(d.format));
        // SAFETY: plain size properties of a fresh descriptor; Metal validates them at creation.
        unsafe {
            td.setWidth(d.width);
            td.setHeight(d.height);
            td.setDepth(if d.dim == TexDim::D3 {
                d.depth_or_layers
            } else {
                1
            });
            td.setArrayLength(if d.dim == TexDim::D2Array {
                d.depth_or_layers
            } else {
                1
            });
            td.setMipmapLevelCount(d.mips);
        }
        let mut usage = MTLTextureUsage::ShaderRead | MTLTextureUsage::PixelFormatView;
        if d.unordered_access {
            usage |= MTLTextureUsage::ShaderWrite;
        }
        if d.render_target {
            usage |= MTLTextureUsage::RenderTarget;
        }
        td.setUsage(usage);
        td.setStorageMode(MTLStorageMode::Shared);
        let tex = self
            .dev
            .newTextureWithDescriptor(&td)
            .ok_or("newTextureWithDescriptor failed")?;
        Ok(Texture { tex })
    }

    fn create_texture_view(&self, t: &Texture, v: &ViewDesc) -> Result<Texture, String> {
        let ty = t.tex.textureType();
        // A depth texture cannot be viewed as a color format (D3D's R32_FLOAT
        // over a depth buffer): it keeps its own, and shaders read it as the
        // float it holds.
        let source = t.tex.pixelFormat();
        let format = if is_depth(source) {
            source
        } else {
            pixel_format(v.format)
        };
        // Metal asserts (it does not return nil) on a range past the texture, which a
        // title view asks for over a texture we made with fewer levels or layers (one
        // over title memory has one level): the range is cut to what is there.
        let fit = |(start, len): (usize, usize), count: usize| {
            (start < count).then(|| NSRange::new(start, len.min(count - start)))
        };
        let slices = if ty == MTLTextureType::Type2DArray {
            t.tex.arrayLength()
        } else {
            1
        };
        let (Some(levels), Some(layers)) =
            (fit(v.mips, t.tex.mipmapLevelCount()), fit(v.slices, slices))
        else {
            return Err(format!(
                "view {:?} {:?} is past the texture",
                v.mips, v.slices
            ));
        };
        // SAFETY: the ranges are inside the texture.
        let tex = unsafe {
            t.tex
                .newTextureViewWithPixelFormat_textureType_levels_slices(format, ty, levels, layers)
        }
        .ok_or("newTextureView refused (format not view-compatible)")?;
        Ok(Texture { tex })
    }

    fn create_texture_over_memory(
        &self,
        buffer: &Buffer,
        title: *const u8,
        d: &TextureDesc,
        row_pitch: usize,
    ) -> Result<Texture, String> {
        // SAFETY: plain sizes; Metal validates them at creation.
        let td = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                pixel_format(d.format),
                d.width,
                d.height,
                false,
            )
        };
        td.setStorageMode(MTLStorageMode::Shared);
        let mut usage = MTLTextureUsage::ShaderRead | MTLTextureUsage::PixelFormatView;
        if d.unordered_access {
            usage |= MTLTextureUsage::ShaderWrite;
        }
        td.setUsage(usage);
        let offset = title as usize - buffer.base as usize;
        let tex = buffer
            .buf
            .newTextureWithDescriptor_offset_bytesPerRow(&td, offset, row_pitch)
            .ok_or("buffer-backed texture refused (offset, pitch or format)")?;
        Ok(Texture { tex })
    }

    fn linear_alignment(&self, f: Format) -> usize {
        self.dev
            .minimumLinearTextureAlignmentForPixelFormat(pixel_format(f))
    }

    fn upload_texture(
        &self,
        t: &Texture,
        mip: usize,
        slice: usize,
        data: &[u8],
        row_pitch: usize,
        slice_pitch: usize,
    ) {
        let (w, h, d) = (
            (t.tex.width() >> mip).max(1),
            (t.tex.height() >> mip).max(1),
            (t.tex.depth() >> mip).max(1),
        );
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: size([w, h, d]),
        };
        // SAFETY: `data` covers the region at the given pitches (callers size it by `Format::row_bytes`).
        unsafe {
            t.tex
                .replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                    region,
                    mip,
                    slice,
                    NonNull::new(data.as_ptr().cast_mut().cast()).unwrap(),
                    row_pitch,
                    slice_pitch,
                )
        };
    }

    fn read_texture(
        &self,
        t: &Texture,
        mip: usize,
        slice: usize,
        out: &mut [u8],
        row_pitch: usize,
        slice_pitch: usize,
    ) {
        let (w, h, d) = (
            (t.tex.width() >> mip).max(1),
            (t.tex.height() >> mip).max(1),
            (t.tex.depth() >> mip).max(1),
        );
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: size([w, h, d]),
        };
        // SAFETY: `out` covers the region at the given pitches.
        unsafe {
            t.tex
                .getBytes_bytesPerRow_bytesPerImage_fromRegion_mipmapLevel_slice(
                    NonNull::new(out.as_mut_ptr().cast()).unwrap(),
                    row_pitch,
                    slice_pitch,
                    region,
                    mip,
                    slice,
                )
        };
    }

    fn create_sampler(&self, d: &SamplerDesc) -> Result<Sampler, String> {
        let sd = MTLSamplerDescriptor::new();
        sd.setMinFilter(filter(d.min_mag));
        sd.setMagFilter(filter(d.min_mag));
        sd.setMipFilter(match d.mip {
            Filter::Nearest => MTLSamplerMipFilter::Nearest,
            Filter::Linear => MTLSamplerMipFilter::Linear,
        });
        sd.setSAddressMode(address(d.address[0]));
        sd.setTAddressMode(address(d.address[1]));
        sd.setRAddressMode(address(d.address[2]));
        sd.setMaxAnisotropy(d.max_anisotropy.max(1));
        // The heap entry stores the sampler's resource ID, which exists only for argument-buffer samplers.
        sd.setSupportArgumentBuffers(true);
        let s = self
            .dev
            .newSamplerStateWithDescriptor(&sd)
            .ok_or("newSamplerState failed")?;
        Ok(Sampler(s))
    }
}

impl GraphicsBackend for Metal {
    type RenderPipeline = RenderPipeline;

    fn create_render_pipeline(&self, d: &RenderPipelineDesc<'_>) -> Result<RenderPipeline, String> {
        let vf = self.stage_function(d.vs, d.vs_layout)?;
        let pd = MTLRenderPipelineDescriptor::new();
        pd.setVertexFunction(Some(&vf));
        if let Some(ps) = d.ps {
            pd.setFragmentFunction(Some(&*self.stage_function(ps, d.ps_layout)?));
        }
        // The converter's stage-in function takes its inputs at attribute indices from
        // `kIRStageInAttributeStartIndex`, in input-signature order; buffers from `kIRVertexBufferBindPoint`.
        let vd = MTLVertexDescriptor::new();
        // SAFETY: indices are below the attribute and buffer limits for sane layouts; plain sizes.
        unsafe {
            for (i, a) in d.attributes.iter().enumerate() {
                let at = vd
                    .attributes()
                    .objectAtIndexedSubscript(STAGE_IN_ATTRIBUTES + i);
                at.setFormat(vertex_format(a.format));
                at.setOffset(a.offset);
                at.setBufferIndex(VERTEX_BUFFERS + a.slot);
            }
            for s in d.streams {
                let l = vd
                    .layouts()
                    .objectAtIndexedSubscript(VERTEX_BUFFERS + s.slot);
                l.setStride(s.stride);
                l.setStepFunction(if s.per_instance {
                    MTLVertexStepFunction::PerInstance
                } else {
                    MTLVertexStepFunction::PerVertex
                });
            }
        }
        pd.setVertexDescriptor(Some(&vd));
        for (i, f) in d.color.iter().enumerate() {
            // SAFETY: colour attachment index below 8.
            let a = unsafe { pd.colorAttachments().objectAtIndexedSubscript(i) };
            a.setPixelFormat(pixel_format(*f));
            let state = d.color_state.get(i).copied().unwrap_or_default();
            let mut mask = MTLColorWriteMask::empty();
            for (bit, channel) in [
                MTLColorWriteMask::Red,
                MTLColorWriteMask::Green,
                MTLColorWriteMask::Blue,
                MTLColorWriteMask::Alpha,
            ]
            .into_iter()
            .enumerate()
            {
                if state.write_mask >> bit & 1 != 0 {
                    mask |= channel;
                }
            }
            a.setWriteMask(mask);
            if let Some(b) = state.blend.filter(|_| !f.is_integer()) {
                a.setBlendingEnabled(true);
                a.setSourceRGBBlendFactor(blend_factor(b.src));
                a.setDestinationRGBBlendFactor(blend_factor(b.dst));
                a.setRgbBlendOperation(blend_op(b.op));
                a.setSourceAlphaBlendFactor(blend_factor(b.src_alpha));
                a.setDestinationAlphaBlendFactor(blend_factor(b.dst_alpha));
                a.setAlphaBlendOperation(blend_op(b.op_alpha));
            }
        }
        let has_stencil = d.depth == Some(Format::D32FloatS8);
        if let Some(f) = d.depth {
            pd.setDepthAttachmentPixelFormat(pixel_format(f));
            if has_stencil {
                pd.setStencilAttachmentPixelFormat(pixel_format(f));
            }
        }
        let pso = self
            .dev
            .newRenderPipelineStateWithDescriptor_error(&pd)
            .map_err(|e| e.localizedDescription().to_string())?;
        let depth = (d.depth_test.is_some() || (d.stencil.is_some() && has_stencil)).then(|| {
            let dd = MTLDepthStencilDescriptor::new();
            let (write, cmp) = d.depth_test.unwrap_or((false, Compare::Always));
            dd.setDepthCompareFunction(compare(cmp));
            dd.setDepthWriteEnabled(write);
            if let Some(s) = d.stencil.filter(|_| has_stencil) {
                dd.setFrontFaceStencil(Some(&stencil_face(&s.front, &s)));
                dd.setBackFaceStencil(Some(&stencil_face(&s.back, &s)));
            }
            self.dev
                .newDepthStencilStateWithDescriptor(&dd)
                .expect("depth state")
        });
        let cull = match d.cull {
            Cull::None => MTLCullMode::None,
            Cull::Front => MTLCullMode::Front,
            Cull::Back => MTLCullMode::Back,
        };
        Ok(RenderPipeline {
            pso,
            depth,
            cull,
            raster: d.raster,
            stencil: has_stencil,
        })
    }

    fn submit_passes(&self, queue: &Queue, work: &PassSubmission<'_, Self>) -> Result<(), String> {
        // Encoder execution status makes a failed command buffer name the
        // encoder that faulted; without it the error is only the code.
        let descriptor = MTLCommandBufferDescriptor::new();
        descriptor.setErrorOptions(MTLCommandBufferErrorOption::EncoderExecutionStatus);
        let cb = queue
            .0
            .commandBufferWithDescriptor(&descriptor)
            .ok_or("no command buffer")?;
        for (f, v) in work.waits {
            cb.encodeWaitForEvent_value(ProtocolObject::from_ref(&*f.0), *v);
        }
        // With the pool set on the queue the encoders do not name the title's buffers, so Metal
        // cannot order them: this command buffer starts after the previous one finished, and each
        // encoder after the one before.
        let set_on_queue = queue.1.get();
        let order = set_on_queue.then(|| {
            let n = queue.3.get();
            queue.3.set(n + 1);
            if n > 0 {
                cb.encodeWaitForEvent_value(&queue.2, n);
            }
            Order {
                fence: self.dev.newFence(),
                started: Cell::new(false),
            }
        });
        let order = order.as_ref().and_then(|o| o.fence.as_ref().map(|f| (&**f, &o.started)));
        for pass in work.passes {
            match pass {
                Pass::Compute(dispatches) => self.encode_compute(&cb, dispatches, set_on_queue, order)?,
                Pass::Render(rp) => self.encode_render(&cb, rp, set_on_queue, order)?,
                Pass::Blit(ops) => self.encode_blit(&cb, ops, order)?,
            }
        }
        if set_on_queue {
            cb.encodeSignalEvent_value(&queue.2, queue.3.get());
        }
        for (f, v) in work.signals {
            cb.encodeSignalEvent_value(ProtocolObject::from_ref(&*f.0), *v);
        }
        // A command buffer the GPU fails is logged (the first few), not silently dropped. The
        // summary makes an out-of-memory failure attributable to the work in the buffer.
        let passes = work.passes.len();
        let (mut dispatches, mut draws, mut blits) = (0usize, 0usize, 0usize);
        for pass in work.passes {
            match pass {
                Pass::Compute(d) => dispatches += d.len(),
                Pass::Render(rp) => draws += rp.draws.len(),
                Pass::Blit(ops) => blits += ops.len(),
            }
        }
        let usage = self.memory_usage();
        let started = std::time::Instant::now();
        let failed = block2::RcBlock::new(move |done: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            // SAFETY: Metal passes the command buffer that completed.
            let done = unsafe { done.as_ref() };
            if let Some(e) = done.error() {
                let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n < 40 {
                    tracing::error!(
                        code = e.code(),
                        error = %e.localizedDescription(),
                        status = ?done.status(),
                        passes,
                        dispatches,
                        draws,
                        blits,
                        allocated = usage.allocated,
                        recommended = usage.recommended,
                        elapsed_ms = started.elapsed().as_millis() as u64,
                        "metal command buffer failed"
                    );
                }
            }
        });
        // SAFETY: the block is retained by Metal until it has run.
        unsafe { cb.addCompletedHandler(&*failed as *const _ as *mut _) };
        cb.commit();
        Ok(())
    }
}

impl Metal {
    fn encode_blit(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        ops: &[BlitOp<'_, Metal>],
        order: Ordering<'_>,
    ) -> Result<(), String> {
        let e = cb.blitCommandEncoder().ok_or("no blit encoder")?;
        if let Some((f, started)) = order
            && started.get()
        {
            e.waitForFence(f);
        }
        let off = |b: &Buffer, p: *const u8| p as usize - b.base as usize;
        for op in ops {
            match op {
                BlitOp::Copy { src, dst, len } => {
                    // SAFETY: both ranges lie inside live buffers (the caller resolved them).
                    unsafe {
                        e.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                            &src.0.buf,
                            off(src.0, src.1),
                            &dst.0.buf,
                            off(dst.0, dst.1),
                            *len,
                        )
                    };
                }
                BlitOp::Fill { dst, len, value } => {
                    let o = off(dst.0, dst.1);
                    e.fillBuffer_range_value(&dst.0.buf, NSRange::new(o, *len), *value);
                }
                BlitOp::CopyTexture { src, dst } => {
                    // SAFETY: both Metal textures are live for the pass.
                    unsafe { e.copyFromTexture_toTexture(&src.tex, &dst.tex) };
                }
                BlitOp::CopyTextureRegion {
                    src,
                    dst,
                    src_slice,
                    src_level,
                    src_origin,
                    size,
                    dst_slice,
                    dst_level,
                    dst_origin,
                } => {
                    // SAFETY: the caller resolved one live texture per side and a
                    // box inside them (D3D12 validates the region before this).
                    unsafe {
                        e.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                            &src.tex,
                            *src_slice as usize,
                            *src_level as usize,
                            MTLOrigin {
                                x: src_origin[0] as usize,
                                y: src_origin[1] as usize,
                                z: src_origin[2] as usize,
                            },
                            MTLSize {
                                width: size[0] as usize,
                                height: size[1] as usize,
                                depth: size[2] as usize,
                            },
                            &dst.tex,
                            *dst_slice as usize,
                            *dst_level as usize,
                            MTLOrigin {
                                x: dst_origin[0] as usize,
                                y: dst_origin[1] as usize,
                                z: dst_origin[2] as usize,
                            },
                        )
                    };
                }
            }
        }
        if let Some((f, started)) = order {
            e.updateFence(f);
            started.set(true);
        }
        e.endEncoding();
        Ok(())
    }
}

impl Metal {
    /// Whether encoder `use` calls are needed for `b`: not when the queue has the pool residency set
    /// and the buffer is in it.
    fn names(&self, b: &Buffer, set_on_queue: bool) -> bool {
        let needed = !(set_on_queue && b.is_resident());
        if needed {
            self.count(|c| c.use_resource += 1);
        }
        needed
    }

    pub(crate) fn encode_compute(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        dispatches: &[crate::backend::Dispatch<'_, Metal>],
        set_on_queue: bool,
        order: Ordering<'_>,
    ) -> Result<(), String> {
        if dispatches.is_empty() {
            return Ok(());
        }
        let e = cb.computeCommandEncoder().ok_or("no compute encoder")?;
        if let Some((f, started)) = order
            && started.get()
        {
            e.waitForFence(f);
        }
        for d in dispatches {
            e.setComputePipelineState(&d.pipeline.0);
            for h in [d.heaps.resource, d.heaps.sampler].into_iter().flatten() {
                self.use_heap_cb(cb, h);
            }
            self.bind_root_compute(&e, &d.root, &d.heaps);
            for b in d.resources.iter().filter(|b| self.names(b, set_on_queue)) {
                e.useResource_usage(
                    ProtocolObject::from_ref(&*b.buf),
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                );
            }
            e.dispatchThreadgroups_threadsPerThreadgroup(size(d.groups), size(d.threads_per_group));
        }
        if let Some((f, started)) = order {
            e.updateFence(f);
            started.set(true);
        }
        e.endEncoding();
        Ok(())
    }

    fn encode_render(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        rp: &RenderPass<'_, Metal>,
        set_on_queue: bool,
        order: Ordering<'_>,
    ) -> Result<(), String> {
        let desc = MTLRenderPassDescriptor::new();
        for (i, c) in rp.color.iter().enumerate() {
            // SAFETY: colour attachment index below 8.
            let a = unsafe { desc.colorAttachments().objectAtIndexedSubscript(i) };
            a.setTexture(Some(&c.texture.tex));
            a.setStoreAction(MTLStoreAction::Store);
            match c.clear {
                Some(v) => {
                    a.setLoadAction(MTLLoadAction::Clear);
                    a.setClearColor(MTLClearColor {
                        red: v[0] as f64,
                        green: v[1] as f64,
                        blue: v[2] as f64,
                        alpha: v[3] as f64,
                    });
                }
                None => a.setLoadAction(MTLLoadAction::Load),
            }
        }
        if let Some(dp) = &rp.depth {
            let a = desc.depthAttachment();
            a.setTexture(Some(&dp.texture.tex));
            a.setStoreAction(MTLStoreAction::Store);
            match dp.clear {
                Some(v) => {
                    a.setLoadAction(MTLLoadAction::Clear);
                    a.setClearDepth(v as f64);
                }
                None => a.setLoadAction(MTLLoadAction::Load),
            }
            // A combined depth and stencil texture is attached as both.
            if dp.texture.tex.pixelFormat() == MTLPixelFormat::Depth32Float_Stencil8 {
                let a = desc.stencilAttachment();
                a.setTexture(Some(&dp.texture.tex));
                a.setStoreAction(MTLStoreAction::Store);
                match dp.clear {
                    Some(_) => {
                        a.setLoadAction(MTLLoadAction::Clear);
                        a.setClearStencil(0);
                    }
                    None => a.setLoadAction(MTLLoadAction::Load),
                }
            }
        }
        let e = cb
            .renderCommandEncoderWithDescriptor(&desc)
            .ok_or("no render encoder")?;
        if let Some((f, started)) = order
            && started.get()
        {
            e.waitForFence_beforeStages(f, MTLRenderStages::Vertex);
        }
        let rt_size = rp
            .color
            .first()
            .map(|c| &c.texture.tex)
            .or(rp.depth.as_ref().map(|d| &d.texture.tex))
            .map_or((0, 0), |t| (t.width(), t.height()));
        for d in &rp.draws {
            for h in [d.heaps.resource, d.heaps.sampler].into_iter().flatten() {
                self.use_heap_cb(cb, h);
            }
            e.setRenderPipelineState(&d.pipeline.pso);
            e.setCullMode(d.pipeline.cull);
            let r = &d.pipeline.raster;
            e.setFrontFacingWinding(if r.front_ccw {
                MTLWinding::CounterClockwise
            } else {
                MTLWinding::Clockwise
            });
            e.setTriangleFillMode(if r.wireframe {
                MTLTriangleFillMode::Lines
            } else {
                MTLTriangleFillMode::Fill
            });
            e.setDepthClipMode(if r.depth_clip {
                MTLDepthClipMode::Clip
            } else {
                MTLDepthClipMode::Clamp
            });
            e.setDepthBias_slopeScale_clamp(r.depth_bias.0, r.depth_bias.1, r.depth_bias.2);
            if let Some(ds) = &d.pipeline.depth {
                e.setDepthStencilState(Some(ds));
            }
            if d.pipeline.stencil {
                e.setStencilReferenceValue(d.state.stencil_ref);
            }
            let [r, g, b, a] = d.state.blend_factor;
            e.setBlendColorRed_green_blue_alpha(r, g, b, a);
            // The target's size bounds the scissor (Metal refuses a rectangle outside it).
            let (tw, th) = rt_size;
            if let Some(v) = d.state.viewport {
                e.setViewport(MTLViewport {
                    originX: v.x as f64,
                    originY: v.y as f64,
                    width: v.width as f64,
                    height: v.height as f64,
                    znear: v.min_depth as f64,
                    zfar: v.max_depth as f64,
                });
            }
            let scissor = match d.state.scissor {
                Some([l, t, r, b]) => {
                    let (l, t) = (l.clamp(0, tw as i32) as usize, t.clamp(0, th as i32) as usize);
                    let (r, b) = (r.clamp(0, tw as i32) as usize, b.clamp(0, th as i32) as usize);
                    MTLScissorRect {
                        x: l,
                        y: t,
                        width: r.saturating_sub(l),
                        height: b.saturating_sub(t),
                    }
                }
                None => MTLScissorRect {
                    x: 0,
                    y: 0,
                    width: tw,
                    height: th,
                },
            };
            if scissor.width == 0 || scissor.height == 0 {
                // Nothing of the target is inside the rectangle: nothing is drawn.
                continue;
            }
            e.setScissorRect(scissor);
            self.bind_root_render(&e, &d.root, d.fragment_root.as_ref(), &d.heaps);
            for b in d.resources.iter().filter(|b| self.names(b, set_on_queue)) {
                e.useResource_usage_stages(
                    ProtocolObject::from_ref(&*b.buf),
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                    MTLRenderStages::Vertex | MTLRenderStages::Fragment,
                );
            }
            for (slot, (b, title)) in d.vertex_buffers.iter().enumerate() {
                let off = *title as usize - b.base as usize;
                // SAFETY: the buffer is live and `off` is inside it.
                unsafe {
                    e.setVertexBuffer_offset_atIndex(Some(&b.buf), off, VERTEX_BUFFERS + slot)
                };
                if self.names(b, set_on_queue) {
                    e.useResource_usage_stages(
                        ProtocolObject::from_ref(&*b.buf),
                        MTLResourceUsage::Read,
                        MTLRenderStages::Vertex,
                    );
                }
            }
            let prim = match d.topology {
                Topology::Triangles => MTLPrimitiveType::Triangle,
                Topology::TriangleStrip => MTLPrimitiveType::TriangleStrip,
                Topology::Lines => MTLPrimitiveType::Line,
                Topology::LineStrip => MTLPrimitiveType::LineStrip,
                Topology::Points => MTLPrimitiveType::Point,
            };
            // What `IRRuntimeDrawPrimitives` / `IRRuntimeDrawIndexedPrimitives` do: draw arguments
            // (`IRRuntimeDrawParams`, 5 words) at slot 4 and the index type at slot 5 (0 = not indexed,
            // else Metal's index type + 1), then the draw itself.
            let mut dp = [0u32; 5];
            let ib: u16;
            match d.call {
                DrawCall::Arrays {
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                } => {
                    dp[..4].copy_from_slice(&[
                        vertex_count,
                        instance_count,
                        first_vertex,
                        first_instance,
                    ]);
                    ib = 0;
                    set_draw_bytes(&e, &dp, ib);
                    // SAFETY: arguments are plain counts.
                    unsafe {
                        e.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                            prim,
                            first_vertex as usize,
                            vertex_count as usize,
                            instance_count as usize,
                            first_instance as usize,
                        )
                    };
                }
                DrawCall::Indexed {
                    index_count,
                    instance_count,
                    first_index,
                    base_vertex,
                    first_instance,
                    wide,
                } => {
                    let (b, title) = d
                        .index_buffer
                        .ok_or("indexed draw without an index buffer")?;
                    let width = if wide { 4 } else { 2 };
                    let off = title as usize - b.base as usize + first_index as usize * width;
                    dp = [
                        index_count,
                        instance_count,
                        off as u32,
                        base_vertex as u32,
                        first_instance,
                    ];
                    ib = if wide { 2 } else { 1 };
                    set_draw_bytes(&e, &dp, ib);
                    if self.names(b, set_on_queue) {
                        e.useResource_usage_stages(
                            ProtocolObject::from_ref(&*b.buf),
                            MTLResourceUsage::Read,
                            MTLRenderStages::Vertex,
                        );
                    }
                    // SAFETY: the index buffer is live and holds `index_count` indices at `off`.
                    unsafe {
                        e.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                            prim,
                            index_count as usize,
                            if wide { MTLIndexType::UInt32 } else { MTLIndexType::UInt16 },
                            &b.buf,
                            off,
                            instance_count as usize,
                            base_vertex as isize,
                            first_instance as usize,
                        )
                    };
                }
            }
        }
        if let Some((f, started)) = order {
            e.updateFence_afterStages(f, MTLRenderStages::Fragment);
            started.set(true);
        }
        e.endEncoding();
        Ok(())
    }
}

fn set_draw_bytes(e: &ProtocolObject<dyn MTLRenderCommandEncoder>, dp: &[u32; 5], index_type: u16) {
    // SAFETY: setVertexBytes copies the bytes before returning.
    unsafe {
        e.setVertexBytes_length_atIndex(
            NonNull::new(dp.as_ptr().cast_mut().cast()).unwrap(),
            20,
            DRAW_ARGUMENTS,
        );
        e.setVertexBytes_length_atIndex(
            NonNull::new((&raw const index_type).cast_mut().cast()).unwrap(),
            2,
            UNIFORMS,
        );
    }
}

/// Whether a pixel format is a depth or stencil one.
fn is_depth(f: MTLPixelFormat) -> bool {
    [
        MTLPixelFormat::Depth16Unorm,
        MTLPixelFormat::Depth32Float,
        MTLPixelFormat::Stencil8,
        MTLPixelFormat::Depth24Unorm_Stencil8,
        MTLPixelFormat::Depth32Float_Stencil8,
    ]
    .contains(&f)
}
