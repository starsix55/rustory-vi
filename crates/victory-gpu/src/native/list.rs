//! A command list on the native host: the calls a title makes on its list (`SetPipelineState`,
//! `SetComputeRoot*`, `Dispatch`, `CopyBufferRegion`, `ClearRenderTargetView`, `DrawInstanced`, ...)
//! recorded as they arrive and turned into backend passes at submission ([`super::host::NativeHost::execute`]).
//!
//! Title addresses are kept as addresses: they resolve to GPU buffers when the list is submitted, by
//! which time the pool observer has told us about every mapping the title made.

use std::rc::Rc;
use victory_backend::backend::{RootLayout, RootParam};
use victory_backend::resources::{DrawCall, DrawState, GraphicsBackend, HeapKind, Topology, Viewport};

/// A heap the title wrote views into: `start` is its descriptor memory (title addresses).
pub struct NativeHeap<B: GraphicsBackend> {
    pub heap: B::Heap,
    pub kind: HeapKind,
    pub start: u64,
    pub count: usize,
    /// View pairs of the pool-offset contract ([`super::addr`]) written into the heap's reserved
    /// entries so far.
    pub pairs: std::cell::Cell<usize>,
    /// The page table built for this heap (its view indices are the heap's own) and the pool-table
    /// generation it is current for.
    pub table: std::cell::RefCell<Option<(u64, Rc<super::addr::Pages<B>>)>>,
    /// The title record (tag, view index, generation) each entry was last mirrored from.
    pub mirrored: std::cell::RefCell<std::collections::HashMap<usize, [u8; 12]>>,
}

/// How a translated title shader's arguments are laid out (`victory_shader::emit::HostShader`):
/// one block of root constants for the user data, then the root constant, shader resource and
/// unordered access views it reads, as addresses. Without it, the title's root parameters go to the
/// shader as they are, in order (shaders compiled for the native host).
#[derive(Clone, Debug, Default)]
pub struct Binding {
    /// Dwords of the user-data block (a multiple of 4).
    pub dwords: usize,
    pub user_data: Vec<victory_shader::emit::UserData>,
    /// Title root parameter indices of the root CBVs, SRVs and UAVs, in this order, with their
    /// kind ([`View`]).
    pub views: Vec<(u8, View)>,
    /// The shader finds title memory through the page table (`t0` of the memory space), which
    /// follows the views in the argument buffer, and the heap's reserved view pairs.
    pub memory: bool,
    /// The pixel stage's own arguments, when the pipeline has a pixel shader that reads any
    /// (a stage has an argument block of its own; this one is the vertex stage's).
    pub ps: Option<Rc<Binding>>,
    /// Some user data takes the heap index of a view of a root view's buffer (and the address's
    /// offset in it): the heap needs the view pairs.
    pub root_views: bool,
    /// The shader's immediate samplers, as [`RootLayout::static_samplers`] declares them.
    pub static_samplers: Vec<[u32; 10]>,
    /// Minimum descriptor-table entries needed per `RootType::Table` root slot: `(slot, count)`.
    pub table_entries: Vec<(u8, usize)>,
}

/// A root view's kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    Cbv,
    Srv,
    Uav,
}

impl Binding {
    pub fn of(shader: &victory_shader::emit::HostShader) -> Binding {
        Binding {
            dwords: shader.user_data_dwords() as usize,
            user_data: shader.user_data.clone(),
            views: shader
                .root_cbvs
                .iter()
                .map(|&r| (r, View::Cbv))
                .chain(shader.root_srvs.iter().map(|&r| (r, View::Srv)))
                .chain(shader.root_uavs.iter().map(|&r| (r, View::Uav)))
                .collect(),
            memory: shader.memory,
            ps: None,
            root_views: shader.user_data.iter().any(|u| {
                u.root == victory_shader::RootType::Descriptor
                    && u.dword >= victory_shader::emit::ROOT_SRV_VIEW
            }),
            static_samplers: shader
                .static_samplers
                .iter()
                .map(|s| {
                    // Anisotropic filtering with a point mip filter becomes linear-mip, as on
                    // D3D12 devices without it.
                    let [u, v, w] = s.address;
                    [s.filter_for_host(false), u, v, w, s.mip_lod_bias.to_bits(), s.max_anisotropy, s.comparison, s.border, s.min_lod.to_bits(), s.max_lod.to_bits()]
                })
                .collect(),
            table_entries: shader
                .table_entries
                .iter()
                .map(|&(slot, count)| (slot, count as usize))
                .collect(),
        }
    }

    /// The root signature the converter lays the argument buffer out by, in the order
    /// `NativeHost::translated_args` fills it: the user-data block, the views, the page table.
    /// `None` when the shader has no arguments (it takes its own signature, if it has one).
    pub fn layout(&self) -> Option<RootLayout> {
        use victory_shader::emit::{MEMORY_SPACE, ROOT_CBV_SPACE, USER_DATA_SPACE};
        let mut params = Vec::new();
        if self.dwords > 0 {
            params.push(RootParam::Constants {
                register: 0,
                space: USER_DATA_SPACE,
                dwords: self.dwords as u32,
            });
        }
        for &(slot, kind) in &self.views {
            let (register, space) = (u32::from(slot), ROOT_CBV_SPACE);
            params.push(match kind {
                View::Cbv => RootParam::Cbv { register, space },
                View::Srv => RootParam::Srv { register, space },
                View::Uav => RootParam::Uav { register, space },
            });
        }
        if self.memory {
            params.push(RootParam::Srv {
                register: 0,
                space: MEMORY_SPACE,
            });
        }
        (!params.is_empty() || !self.static_samplers.is_empty()).then(|| RootLayout {
            params,
            // Memory no mapped pool holds is read through a view of size 0.
            bounds_checked: true,
            static_samplers: self.static_samplers.clone(),
        })
    }
}

/// A compute pipeline and the group size its shader was written with (`numthreads`).
pub struct ComputePso<B: GraphicsBackend> {
    pub pso: B::Pipeline,
    pub threads: [usize; 3],
    pub binding: Option<Binding>,
}

/// One root parameter as the title set it.
pub enum Root<B: GraphicsBackend> {
    Constants(Vec<u32>),
    /// A root SRV, UAV or CBV: a title address.
    View(u64),
    /// A descriptor table: heap and the index of its first entry.
    Table(Rc<NativeHeap<B>>, usize),
}

impl<B: GraphicsBackend> Clone for Root<B> {
    fn clone(&self) -> Self {
        match self {
            Root::Constants(c) => Root::Constants(c.clone()),
            Root::View(a) => Root::View(*a),
            Root::Table(h, i) => Root::Table(h.clone(), *i),
        }
    }
}

type Heaps<B> = (Option<Rc<NativeHeap<B>>>, Option<Rc<NativeHeap<B>>>);

pub struct Targets<B: GraphicsBackend> {
    pub color: Vec<Rc<B::Texture>>,
    pub depth: Option<Rc<B::Texture>>,
}

impl<B: GraphicsBackend> Clone for Targets<B> {
    fn clone(&self) -> Self {
        Targets {
            color: self.color.clone(),
            depth: self.depth.clone(),
        }
    }
}

impl<B: GraphicsBackend> Targets<B> {
    pub(super) fn same(&self, o: &Targets<B>) -> bool {
        self.color.len() == o.color.len()
            && self
                .color
                .iter()
                .zip(&o.color)
                .all(|(a, b)| Rc::ptr_eq(a, b))
            && match (&self.depth, &o.depth) {
                (Some(a), Some(b)) => Rc::ptr_eq(a, b),
                (None, None) => true,
                _ => false,
            }
    }
}

pub enum Cmd<B: GraphicsBackend> {
    Dispatch {
        pso: Rc<ComputePso<B>>,
        root: Vec<Option<Root<B>>>,
        heaps: Heaps<B>,
        groups: [usize; 3],
    },
    Copy {
        dst: u64,
        src: u64,
        len: usize,
    },
    Fill {
        dst: u64,
        len: usize,
        value: u8,
    },
    /// `CopyResource` between textures.
    CopyTexture {
        dst: usize,
        src: usize,
    },
    /// `CopyTextureRegion` between textures.
    CopyTextureRegion {
        dst: usize,
        src: usize,
        dst_slice: u32,
        dst_level: u32,
        dst_origin: [u32; 3],
        src_slice: u32,
        src_level: u32,
        src_origin: [u32; 3],
        size: [u32; 3],
    },
    Clear {
        color: Option<(Rc<B::Texture>, [f32; 4])>,
        depth: Option<(Rc<B::Texture>, f32)>,
    },
    Draw {
        pso: Rc<B::RenderPipeline>,
        binding: Option<Rc<Binding>>,
        topology: Topology,
        root: Vec<Option<Root<B>>>,
        heaps: Heaps<B>,
        targets: Targets<B>,
        vertex: Vec<u64>,
        index: Option<(u64, bool)>,
        state: DrawState,
        call: DrawCall,
    },
}

impl<B: GraphicsBackend> Clone for Cmd<B> {
    fn clone(&self) -> Self {
        match self {
            Cmd::Dispatch {
                pso,
                root,
                heaps,
                groups,
            } => Cmd::Dispatch {
                pso: pso.clone(),
                root: root.clone(),
                heaps: heaps.clone(),
                groups: *groups,
            },
            Cmd::Copy { dst, src, len } => Cmd::Copy {
                dst: *dst,
                src: *src,
                len: *len,
            },
            Cmd::Fill { dst, len, value } => Cmd::Fill {
                dst: *dst,
                len: *len,
                value: *value,
            },
            Cmd::CopyTexture { dst, src } => Cmd::CopyTexture {
                dst: *dst,
                src: *src,
            },
            Cmd::CopyTextureRegion {
                dst,
                src,
                dst_slice,
                dst_level,
                dst_origin,
                src_slice,
                src_level,
                src_origin,
                size,
            } => Cmd::CopyTextureRegion {
                dst: *dst,
                src: *src,
                dst_slice: *dst_slice,
                dst_level: *dst_level,
                dst_origin: *dst_origin,
                src_slice: *src_slice,
                src_level: *src_level,
                src_origin: *src_origin,
                size: *size,
            },
            Cmd::Clear { color, depth } => Cmd::Clear {
                color: color.clone(),
                depth: depth.clone(),
            },
            Cmd::Draw {
                pso,
                binding,
                topology,
                root,
                heaps,
                targets,
                vertex,
                index,
                state,
                call,
            } => Cmd::Draw {
                pso: pso.clone(),
                binding: binding.clone(),
                topology: *topology,
                root: root.clone(),
                heaps: heaps.clone(),
                targets: targets.clone(),
                vertex: vertex.clone(),
                index: *index,
                state: *state,
                call: *call,
            },
        }
    }
}

/// The state a title sets between commands.
struct State<B: GraphicsBackend> {
    compute: Option<Rc<ComputePso<B>>>,
    graphics: Option<(Rc<B::RenderPipeline>, Topology)>,
    graphics_binding: Option<Rc<Binding>>,
    compute_root: Vec<Option<Root<B>>>,
    graphics_root: Vec<Option<Root<B>>>,
    heaps: Heaps<B>,
    targets: Targets<B>,
    vertex: Vec<u64>,
    index: Option<(u64, bool)>,
    draw: DrawState,
}

pub struct NativeList<B: GraphicsBackend> {
    pub(super) cmds: Vec<Cmd<B>>,
    state: State<B>,
}

/// D3D12's limits, which the title's root signature obeys: a slot or constant offset past these is
/// a misread command, not a request to grow a host table. Growing it was able to reserve
/// `u32::MAX` entries (Native-Victory #15) and exhaust command-buffer memory.
const MAX_ROOT_SLOTS: usize = 64;
const MAX_ROOT_CONSTANTS: usize = 64;
const MAX_VERTEX_SLOTS: usize = 32;

fn out_of_range(what: &str, value: usize, max: usize) {
    static WARNED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if WARNED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 20 {
        tracing::warn!(what, value, max, "a title set a root parameter past D3D12's limit; ignored");
    }
}

fn set<B: GraphicsBackend>(root: &mut Vec<Option<Root<B>>>, slot: usize, value: Root<B>) {
    if slot >= MAX_ROOT_SLOTS {
        out_of_range("root slot", slot, MAX_ROOT_SLOTS);
        return;
    }
    if let Root::Constants(words) = &value
        && words.len() > MAX_ROOT_CONSTANTS
    {
        out_of_range("root constants", words.len(), MAX_ROOT_CONSTANTS);
        return;
    }
    if root.len() <= slot {
        root.resize_with(slot + 1, || None);
    }
    root[slot] = Some(value);
}

impl<B: GraphicsBackend> NativeList<B> {
    /// `CopyResource` between textures; resolved at submission.
    pub fn copy_texture(&mut self, dst: usize, src: usize) {
        if std::env::var_os("VICTORY_TRACE_TEXTURE_COPIES").is_some() {
            tracing::info!(dst = format!("{dst:#x}"), src = format!("{src:#x}"), "texture copy recorded");
        }
        self.cmds.push(Cmd::CopyTexture { dst, src });
    }

    /// `CopyTextureRegion` between textures; resolved at submission.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_texture_region(
        &mut self,
        dst: usize,
        src: usize,
        dst_slice: u32,
        dst_level: u32,
        dst_origin: [u32; 3],
        src_slice: u32,
        src_level: u32,
        src_origin: [u32; 3],
        size: [u32; 3],
    ) {
        if std::env::var_os("VICTORY_TRACE_TEXTURE_COPIES").is_some() {
            tracing::info!(dst = format!("{dst:#x}"), src = format!("{src:#x}"), ?size, "texture region copy recorded");
        }
        self.cmds.push(Cmd::CopyTextureRegion {
            dst,
            src,
            dst_slice,
            dst_level,
            dst_origin,
            src_slice,
            src_level,
            src_origin,
            size,
        });
    }
}

impl<B: GraphicsBackend> Default for NativeList<B> {
    fn default() -> Self {
        NativeList {
            cmds: Vec::new(),
            state: State {
                compute: None,
                graphics: None,
                graphics_binding: None,
                compute_root: Vec::new(),
                graphics_root: Vec::new(),
                heaps: (None, None),
                targets: Targets {
                    color: Vec::new(),
                    depth: None,
                },
                vertex: Vec::new(),
                index: None,
                draw: DrawState::default(),
            },
        }
    }
}

impl<B: GraphicsBackend> NativeList<B> {
    /// `Reset`: forget everything recorded and set.
    pub fn reset(&mut self) {
        *self = NativeList::default();
    }

    pub fn is_empty(&self) -> bool {
        self.cmds.is_empty()
    }

    /// What was recorded so far as a list of its own, leaving this one empty with its state kept
    /// (the title goes on recording after a wait with the pipeline and root arguments it had set).
    pub fn cut(&mut self) -> NativeList<B> {
        NativeList {
            cmds: std::mem::take(&mut self.cmds),
            ..NativeList::default()
        }
    }

    /// A copy of the recorded commands (a title may reset and re-record a list it submitted).
    pub fn snapshot(&self) -> NativeList<B> {
        NativeList {
            cmds: self.cmds.clone(),
            ..NativeList::default()
        }
    }

    /// A write of `value`'s low `bytes` (little endian) at title address `at`, in order with the
    /// rest of the list on the GPU (bottom-of-pipe value writes): byte fills.
    pub fn write_value(&mut self, at: u64, value: u64, bytes: usize) {
        for (i, b) in value.to_le_bytes().iter().take(bytes).enumerate() {
            self.fill(at + i as u64, 1, *b);
        }
    }

    /// `SetDescriptorHeaps`.
    pub fn set_heaps(
        &mut self,
        resource: Option<&Rc<NativeHeap<B>>>,
        sampler: Option<&Rc<NativeHeap<B>>>,
    ) {
        self.state.heaps = (resource.cloned(), sampler.cloned());
    }

    pub fn set_compute_pipeline(&mut self, pso: &Rc<ComputePso<B>>) {
        self.state.compute = Some(pso.clone());
    }

    pub fn set_graphics_pipeline(&mut self, pso: &Rc<B::RenderPipeline>, topology: Topology) {
        self.state.graphics = Some((pso.clone(), topology));
        self.state.graphics_binding = None;
    }

    /// The argument layout of the graphics pipeline set last (a translated title shader's).
    pub fn set_graphics_binding(&mut self, binding: Option<Rc<Binding>>) {
        self.state.graphics_binding = binding;
    }

    /// `SetComputeRoot32BitConstants` (the whole parameter).
    pub fn set_compute_root_constants(&mut self, slot: usize, words: &[u32]) {
        set(
            &mut self.state.compute_root,
            slot,
            Root::Constants(words.to_vec()),
        );
    }

    /// `SetComputeRoot32BitConstant(s)` / `SetGraphicsRoot32BitConstant(s)` at `offset` words into
    /// the parameter: the words before it keep what was set (zero if nothing was).
    pub fn set_root_constants_at(&mut self, compute: bool, slot: usize, offset: usize, words: &[u32]) {
        if slot >= MAX_ROOT_SLOTS
            || offset
                .checked_add(words.len())
                .is_none_or(|end| end > MAX_ROOT_CONSTANTS)
        {
            out_of_range(
                "root constant offset",
                offset.saturating_add(words.len()),
                MAX_ROOT_CONSTANTS,
            );
            return;
        }
        let root = if compute {
            &mut self.state.compute_root
        } else {
            &mut self.state.graphics_root
        };
        let mut all = match root.get(slot) {
            Some(Some(Root::Constants(c))) => c.clone(),
            _ => Vec::new(),
        };
        if all.len() < offset + words.len() {
            all.resize(offset + words.len(), 0);
        }
        all[offset..offset + words.len()].copy_from_slice(words);
        set(root, slot, Root::Constants(all));
    }

    /// `SetComputeRootShaderResourceView` / `UnorderedAccessView` / `ConstantBufferView`.
    pub fn set_compute_root_view(&mut self, slot: usize, address: u64) {
        set(&mut self.state.compute_root, slot, Root::View(address));
    }

    /// `SetComputeRootDescriptorTable`.
    pub fn set_compute_root_table(&mut self, slot: usize, heap: &Rc<NativeHeap<B>>, index: usize) {
        set(
            &mut self.state.compute_root,
            slot,
            Root::Table(heap.clone(), index),
        );
    }

    pub fn set_graphics_root_constants(&mut self, slot: usize, words: &[u32]) {
        set(
            &mut self.state.graphics_root,
            slot,
            Root::Constants(words.to_vec()),
        );
    }

    pub fn set_graphics_root_view(&mut self, slot: usize, address: u64) {
        set(&mut self.state.graphics_root, slot, Root::View(address));
    }

    pub fn set_graphics_root_table(&mut self, slot: usize, heap: &Rc<NativeHeap<B>>, index: usize) {
        set(
            &mut self.state.graphics_root,
            slot,
            Root::Table(heap.clone(), index),
        );
    }

    /// `Dispatch`. Needs the compute pipeline set.
    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) -> Result<(), String> {
        let pso = self
            .state
            .compute
            .clone()
            .ok_or("dispatch without a compute pipeline")?;
        self.cmds.push(Cmd::Dispatch {
            pso,
            root: self.state.compute_root.clone(),
            heaps: self.state.heaps.clone(),
            groups: [x as usize, y as usize, z as usize],
        });
        Ok(())
    }

    /// `CopyBufferRegion` between title addresses. Runs on the GPU, in order with the rest.
    pub fn copy_buffer_region(&mut self, dst: u64, src: u64, len: usize) {
        self.cmds.push(Cmd::Copy { dst, src, len });
    }

    /// A byte fill of title memory (the console's `FillMemoryX`).
    pub fn fill(&mut self, dst: u64, len: usize, value: u8) {
        self.cmds.push(Cmd::Fill { dst, len, value });
    }

    /// `OMSetRenderTargets`.
    pub fn set_render_targets(&mut self, color: &[Rc<B::Texture>], depth: Option<&Rc<B::Texture>>) {
        self.state.targets = Targets {
            color: color.to_vec(),
            depth: depth.cloned(),
        };
    }

    /// `ClearRenderTargetView`. shortcut: each clear is a render pass of its own (load action
    /// clear); folding it into the following draws' pass saves one pass per clear.
    pub fn clear_render_target(&mut self, target: &Rc<B::Texture>, color: [f32; 4]) {
        self.cmds.push(Cmd::Clear {
            color: Some((target.clone(), color)),
            depth: None,
        });
    }

    pub fn clear_depth(&mut self, target: &Rc<B::Texture>, depth: f32) {
        self.cmds.push(Cmd::Clear {
            color: None,
            depth: Some((target.clone(), depth)),
        });
    }

    /// `RSSetViewports` (the first viewport; the host draws into one).
    pub fn set_viewport(&mut self, viewport: Option<Viewport>) {
        self.state.draw.viewport = viewport;
    }

    /// `RSSetScissorRects` (the first rectangle): left, top, right, bottom.
    pub fn set_scissor(&mut self, rect: Option<[i32; 4]>) {
        self.state.draw.scissor = rect;
    }

    /// `OMSetStencilRef`.
    pub fn set_stencil_ref(&mut self, reference: u32) {
        self.state.draw.stencil_ref = reference;
    }

    /// `OMSetBlendFactor`.
    pub fn set_blend_factor(&mut self, factor: [f32; 4]) {
        self.state.draw.blend_factor = factor;
    }

    /// `IASetVertexBuffers` for one slot: the title address of its data (the stride is the pipeline's).
    pub fn set_vertex_buffer(&mut self, slot: usize, address: u64) {
        if slot >= MAX_VERTEX_SLOTS {
            out_of_range("vertex buffer slot", slot, MAX_VERTEX_SLOTS);
            return;
        }
        if self.state.vertex.len() <= slot {
            self.state.vertex.resize(slot + 1, 0);
        }
        self.state.vertex[slot] = address;
    }

    /// The address bound as vertex buffer `slot`, if any (for the target trace).
    pub fn vertex_buffer_address(&self, slot: usize) -> Option<u64> {
        self.state.vertex.get(slot).copied()
    }

    pub fn set_index_buffer(&mut self, address: u64, wide: bool) {
        self.state.index = Some((address, wide));
    }

    pub fn clear_index_buffer(&mut self) {
        self.state.index = None;
    }

    fn draw_call(&mut self, call: DrawCall) -> Result<(), String> {
        let (pso, topology) = self
            .state
            .graphics
            .clone()
            .ok_or("draw without a graphics pipeline")?;
        self.cmds.push(Cmd::Draw {
            pso,
            binding: self.state.graphics_binding.clone(),
            topology,
            root: self.state.graphics_root.clone(),
            heaps: self.state.heaps.clone(),
            targets: self.state.targets.clone(),
            vertex: self.state.vertex.clone(),
            index: self.state.index,
            state: self.state.draw,
            call,
        });
        Ok(())
    }

    pub fn draw_instanced(
        &mut self,
        vertices: u32,
        instances: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Result<(), String> {
        self.draw_call(DrawCall::Arrays {
            vertex_count: vertices,
            instance_count: instances,
            first_vertex,
            first_instance,
        })
    }

    pub fn draw_indexed_instanced(
        &mut self,
        indices: u32,
        instances: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) -> Result<(), String> {
        let wide = self
            .state
            .index
            .ok_or("indexed draw without an index buffer")?
            .1;
        self.draw_call(DrawCall::Indexed {
            index_count: indices,
            instance_count: instances,
            first_index,
            base_vertex,
            first_instance,
            wide,
        })
    }
}

#[cfg(all(test, target_os = "macos"))]
mod limit_tests {
    use super::*;
    use victory_metal::metal::Metal;

    /// Native-Victory #15: a misread root slot used to resize the table to the
    /// slot number, which could reserve `u32::MAX` entries.
    #[test]
    fn a_root_slot_past_d3d12s_limit_does_not_grow_the_table() {
        let mut root = Vec::new();
        set::<Metal>(&mut root, u32::MAX as usize, Root::View(0));
        assert!(root.is_empty());
        set::<Metal>(&mut root, MAX_ROOT_SLOTS - 1, Root::View(0));
        assert_eq!(root.len(), MAX_ROOT_SLOTS);
        assert!(root[MAX_ROOT_SLOTS - 1].is_some());
    }

    #[test]
    fn root_constants_past_d3d12s_limit_are_ignored() {
        let mut root = Vec::new();
        set::<Metal>(&mut root, 2, Root::Constants(vec![0; MAX_ROOT_CONSTANTS + 1]));
        assert!(root.is_empty());
        set::<Metal>(&mut root, 2, Root::Constants(vec![0; MAX_ROOT_CONSTANTS]));
        assert_eq!(root.len(), 3);
    }
}
