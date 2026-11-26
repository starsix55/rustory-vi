//! victory-gpu on Metal, end to end, with command streams of our own (the HLSL under
//! `victory-metal/tests/fixtures`, our own data): the calls a title list receives, recorded into a
//! [`NativeList`], submitted through [`Host`], read back from title memory. Title memory is a pool
//! (`Alias::primary`, what the GPU buffer wraps) shown to the "title" at a second address
//! (`Alias::alias`) through the pool table, so every address the list names is translated.
//! Skipped when Apple's Metal Shader Converter is not installed.

use super::Host;
use super::glue::with_title;
use super::list::NativeList;
use super::memory::{PoolInfo, PoolObserver};
use super::title::Title;
use std::rc::Rc;
use victory_backend::backend::ComputeBackend;
use victory_backend::resources::*;
use victory_formats::texture::Swizzle;
use victory_metal::mem::{Alias, page_size};
use victory_metal::metal::Metal;

#[path = "addr_tests.rs"]
mod addr_tests;
#[path = "shader_tests.rs"]
mod shader_tests;
#[path = "state_tests.rs"]
mod state_tests;
#[path = "texture_tests.rs"]
mod texture_tests;
#[path = "title_tests.rs"]
mod title_tests;

fn host() -> Option<Host> {
    if !std::path::Path::new(victory_metal::msc::DEFAULT_DYLIB).exists()
        && std::env::var_os("VICTORY_MSC_DYLIB").is_none()
    {
        eprintln!("skipped: Metal Shader Converter not installed");
        return None;
    }
    Some(Host::new(Metal::new().unwrap()))
}

fn fixture(name: &str) -> Vec<u8> {
    let p = format!(
        "{}/../victory-metal/tests/fixtures/{name}.dxil",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(&p).unwrap_or_else(|e| panic!("{p}: {e}"))
}

/// A pool of `pages` pages that the title sees at `alias`; GPU buffers wrap `primary`.
struct Pool {
    mem: Alias,
}

impl Pool {
    fn new(h: &Host, pages: usize) -> Pool {
        // Tests assume at least 16 KiB per page (e.g. offsets 16384, 32768); under x86_64
        // Rosetta `page_size()` reports 4 KiB, so clamp the unit to at least 16 KiB.
        let mem = Alias::new(pages * page_size().max(16384)).unwrap();
        let info = PoolInfo {
            id: 1,
            base: mem.primary as usize,
            len: mem.len,
            memory_entry: None,
        };
        h.memory().pool_created(&info);
        h.memory().mapped(mem.alias as usize, mem.len, &info, 0);
        Pool { mem }
    }

    /// The title's address of byte `at`.
    fn va(&self, at: usize) -> u64 {
        self.mem.alias as u64 + at as u64
    }

    fn write<T: Copy>(&self, at: usize, v: &[T]) {
        // SAFETY: inside the mapping; the GPU is idle while the test writes.
        unsafe {
            std::ptr::copy_nonoverlapping(v.as_ptr(), self.mem.alias.add(at).cast(), v.len())
        };
    }

    fn read<T: Copy>(&self, at: usize, n: usize) -> Vec<T> {
        // SAFETY: inside the mapping; read after the fence wait.
        unsafe { std::slice::from_raw_parts(self.mem.alias.add(at).cast::<T>(), n).to_vec() }
    }
}

fn run(h: &Host, lists: &[&NativeList<Metal>]) {
    let q = h.backend.create_queue().unwrap();
    let f = h.backend.create_fence(0).unwrap();
    h.execute(&q, lists, &[], &[(&f, 1)]).unwrap();
    assert!(h.backend.wait(&f, 1, 10_000), "GPU did not finish");
}

const N: usize = 1024;

/// dst[i] = src[i] * 5 + 7, root views to title memory; the source arrives through a GPU copy and
/// the destination is cleared first with a GPU fill, all in one list.
#[test]
fn compute_dispatch_writes_title_memory() {
    let Some(h) = host() else { return };
    let pool = Pool::new(&h, 4);
    let (staged, src, dst) = (0, 16384, 32768);
    pool.write(staged, &(0..N as u32).collect::<Vec<_>>());
    pool.write(dst, &[0xEEEE_EEEEu32; N]);
    let pso = h.compute_pipeline(&fixture_affine(), [64, 1, 1]).unwrap();

    let mut l = NativeList::default();
    l.copy_buffer_region(pool.va(src), pool.va(staged), N * 4);
    l.fill(pool.va(dst), N * 4, 0);
    l.set_compute_pipeline(&pso);
    l.set_compute_root_constants(0, &[5, 7]);
    l.set_compute_root_view(1, pool.va(src));
    l.set_compute_root_view(2, pool.va(dst));
    l.dispatch((N / 64) as u32, 1, 1).unwrap();
    run(&h, &[&l]);

    let out: Vec<u32> = pool.read(dst, N);
    assert!(
        out.iter().enumerate().all(|(i, v)| *v == i as u32 * 5 + 7),
        "{:?}",
        &out[..8]
    );
}

fn fixture_affine() -> Vec<u8> {
    let p = format!(
        "{}/../victory-metal/tests/affine.dxil",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(p).unwrap()
}

/// Descriptors the way a title writes them: `CreateShaderResourceView`-family device calls leave a
/// record in 32-byte heap slots, which `mirror` decodes into the backend's heap; the dispatch finds
/// its buffers through a descriptor table. Two submissions on one queue, the second waiting on the
/// first's fence value (no CPU wait between).
#[test]
fn descriptor_table_buffers_decode_into_metal_heap() {
    let Some(h) = host() else { return };
    let pool = Pool::new(&h, 4);
    let (src, dst) = (0, 16384);
    pool.write(src, &(0..64u32).collect::<Vec<_>>());
    let slots = vec![0u32; 8 * 8]; // 8 views of 32 bytes: the title's descriptor memory
    let start = slots.as_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, start, 8).unwrap();

    // CreateShaderResourceView / UnorderedAccessView of memory the title placed itself (raw buffers).
    let placed = |at: usize| {
        let mut d = [0u32; 16];
        d[1] = 1; // buffer view
        d[6] = 64; // elements
        d[8] = 1; // raw (SRV flags word)
        d[10] = pool.va(at) as u32;
        d[11] = (pool.va(at) >> 32) as u32;
        d
    };
    let mut resource = [0u32; 14];
    resource[0] = 1; // buffer
    crate::descriptor::create_placed_srv(std::ptr::null_mut(), &resource, &placed(src), start);
    let mut uav = placed(dst);
    (uav[4], uav[6]) = (64, 0); // UAV words: element count at 4, stride at 5, flags at 8
    crate::descriptor::create_placed_uav(std::ptr::null_mut(), &resource, &uav, start + 32);
    assert!(h.mirror(start).unwrap() && h.mirror(start + 32).unwrap());

    let pso = h.compute_pipeline(&fixture("heap_rw"), [64, 1, 1]).unwrap();
    let mut l = NativeList::default();
    l.set_heaps(Some(&heap), None);
    l.set_compute_pipeline(&pso);
    l.set_compute_root_constants(0, &[3, 1]);
    l.set_compute_root_table(1, &heap, 0);
    l.dispatch(1, 1, 1).unwrap();
    // A second list on the same queue reuses the output as its input through a root view.
    let aff = h.compute_pipeline(&fixture_affine(), [64, 1, 1]).unwrap();
    let mut l2 = NativeList::default();
    l2.set_compute_pipeline(&aff);
    l2.set_compute_root_constants(0, &[2, 0]);
    l2.set_compute_root_view(1, pool.va(dst));
    l2.set_compute_root_view(2, pool.va(dst + 4096));
    l2.dispatch(1, 1, 1).unwrap();

    let q = h.backend.create_queue().unwrap();
    let (a, b) = (
        h.backend.create_fence(0).unwrap(),
        h.backend.create_fence(0).unwrap(),
    );
    h.execute(&q, &[&l], &[], &[(&a, 1)]).unwrap();
    h.execute(&q, &[&l2], &[(&a, 1)], &[(&b, 1)]).unwrap();
    assert!(h.backend.wait(&b, 1, 10_000));
    assert_eq!(
        pool.read::<u32>(dst, 64),
        (0..64).map(|i| i * 3 + 1).collect::<Vec<_>>()
    );
    assert_eq!(
        pool.read::<u32>(dst + 4096, 64),
        (0..64).map(|i| (i * 3 + 1) * 2).collect::<Vec<_>>()
    );
    drop(slots);
}

/// A title copies view records between heap slots with plain stores (no device call): the
/// submission mirrors a bound table's entries whose record changed.
#[test]
fn table_records_copied_by_the_title_are_mirrored_at_submission() {
    let Some(h) = host() else { return };
    let pool = Pool::new(&h, 4);
    let (src, dst) = (0, 16384);
    pool.write(src, &(0..64u32).collect::<Vec<_>>());
    let mut slots = vec![0u32; 8 * 8];
    let start = slots.as_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, start, 8).unwrap();
    let placed = |at: usize| {
        let mut d = [0u32; 16];
        (d[1], d[6], d[8]) = (1, 64, 1); // raw buffer view of 64 elements
        d[10] = pool.va(at) as u32;
        d[11] = (pool.va(at) >> 32) as u32;
        d
    };
    let mut resource = [0u32; 14];
    resource[0] = 1;
    // Made in slots 4 and 5 (mirrored there), then copied by the title to 0 and 1.
    crate::descriptor::create_placed_srv(std::ptr::null_mut(), &resource, &placed(src), start + 4 * 32);
    let mut uav = placed(dst);
    (uav[4], uav[6]) = (64, 0);
    crate::descriptor::create_placed_uav(std::ptr::null_mut(), &resource, &uav, start + 5 * 32);
    assert!(h.mirror(start + 4 * 32).unwrap() && h.mirror(start + 5 * 32).unwrap());
    slots.copy_within(4 * 8..6 * 8, 0);

    let pso = h.compute_pipeline(&fixture("heap_rw"), [64, 1, 1]).unwrap();
    let mut l = NativeList::default();
    l.set_heaps(Some(&heap), None);
    l.set_compute_pipeline(&pso);
    l.set_compute_root_constants(0, &[3, 1]);
    l.set_compute_root_table(1, &heap, 0);
    l.dispatch(1, 1, 1).unwrap();
    let q = h.backend.create_queue().unwrap();
    let done = h.backend.create_fence(0).unwrap();
    h.execute(&q, &[&l], &[], &[(&done, 1)]).unwrap();
    assert!(h.backend.wait(&done, 1, 10_000));
    assert_eq!(pool.read::<u32>(dst, 64), (0..64).map(|i| i * 3 + 1).collect::<Vec<_>>());
    drop(slots);
}

type Vertex = [f32; 8];

fn vertex(x: f32, y: f32, c: [f32; 4]) -> Vertex {
    [x, y, 0.5, 0.5, c[0], c[1], c[2], c[3]]
}

/// Clear a render target, draw a triangle with a texture and sampler reached through descriptor
/// tables (written the title's way, then decoded), and read the pixels back.
#[test]
fn clear_and_draw_into_render_target() {
    let Some(h) = host() else { return };
    const W: usize = 16;
    let pool = Pool::new(&h, 2);
    pool.write(
        0,
        &[
            vertex(-0.5, -0.5, [1.0, 0.0, 0.0, 1.0]),
            vertex(0.5, -0.5, [1.0, 0.0, 0.0, 1.0]),
            vertex(0.0, 0.5, [1.0, 0.0, 0.0, 1.0]),
        ],
    );

    let attrs = [
        VertexAttr {
            format: VertexFormat::Float2,
            slot: 0,
            offset: 0,
        },
        VertexAttr {
            format: VertexFormat::Float2,
            slot: 0,
            offset: 8,
        },
        VertexAttr {
            format: VertexFormat::Float4,
            slot: 0,
            offset: 16,
        },
    ];
    let pso = h
        .render_pipeline(&RenderPipelineDesc {
            vs: &fixture("tri_vs"),
            ps: Some(&fixture("tri_ps")),
            vs_layout: None,
            ps_layout: None,
            attributes: &attrs,
            streams: &[VertexStream {
                slot: 0,
                stride: 32,
                per_instance: false,
            }],
            color: &[Format::Rgba8Unorm],
            depth: None,
            depth_test: None,
            cull: Cull::None,
            ..Default::default()
        })
        .unwrap();
    let mut rt = TextureDesc::d2(Format::Rgba8Unorm, W, W);
    rt.render_target = true;
    let color = h.texture(&rt).unwrap();
    let white = h
        .texture(&TextureDesc::d2(Format::Rgba8Unorm, 1, 1))
        .unwrap();
    h.backend.upload_texture(&white, 0, 0, &[255; 4], 4, 0);
    h.register_texture(0xA11CE, &white);

    let views = vec![0u32; 8 * 4];
    let smp = vec![0u32; 4 * 4];
    let (vstart, sstart) = (views.as_ptr() as u64, smp.as_ptr() as u64);
    let vheap = h.heap(HeapKind::Resource, vstart, 4).unwrap();
    let sheap = h.heap(HeapKind::Sampler, sstart, 4).unwrap();
    // CreateShaderResourceView(texture, Texture2D, mips 0..1) and CreateSamplerX (bilinear, clamp).
    let mut srv = [0u32; 10];
    (srv[0], srv[1], srv[5]) = (28, 4, 1);
    crate::descriptor::create_srv(std::ptr::null_mut(), 0xA11CE as _, &srv, vstart);
    let mut sw = [0u32; 27];
    (sw[1], sw[2], sw[6], sw[7], sw[8]) = (1, 1, 3, 3, 3);
    crate::descriptor::create_sampler_x(std::ptr::null_mut(), &sw, sstart);
    assert!(h.mirror(vstart).unwrap() && h.mirror(sstart).unwrap());
    assert_eq!(
        h.heap_texture_refs().get(&0xA11CE),
        Some(&1),
        "the heap entry pins the texture"
    );

    let mut l = NativeList::default();
    l.set_render_targets(std::slice::from_ref(&color), None);
    l.clear_render_target(&color, [0.0, 0.0, 1.0, 1.0]);
    l.set_graphics_pipeline(&pso, Topology::Triangles);
    l.set_heaps(Some(&vheap), Some(&sheap));
    l.set_graphics_root_constants(0, &[0.5f32.to_bits(), 0, 0, 0]);
    l.set_graphics_root_table(1, &vheap, 0);
    l.set_graphics_root_table(2, &sheap, 0);
    l.set_vertex_buffer(0, pool.va(0));
    l.draw_instanced(3, 1, 0, 0).unwrap();
    run(&h, &[&l]);

    let mut out = vec![0u8; W * W * 4];
    h.backend.read_texture(&color, 0, 0, &mut out, W * 4, 0);
    let px = |x: usize, y: usize| <[u8; 4]>::try_from(&out[(y * W + x) * 4..][..4]).unwrap();
    assert_eq!(px(8, 8), [255, 0, 0, 255], "inside the triangle");
    assert_eq!(px(0, 0), [0, 0, 255, 255], "cleared corner");
    assert_eq!(px(15, 15), [0, 0, 255, 255], "cleared corner");
    let _ = Rc::strong_count(&color);
    h.drop_heap(vstart);
    assert!(
        h.heap_texture_refs().is_empty(),
        "dropping the heap releases its texture references"
    );
    drop((views, smp));
}

/// An address outside every mapped pool is an error at submission, not a GPU fault.
#[test]
fn unmapped_title_address_is_refused() {
    let Some(h) = host() else { return };
    let pso = h.compute_pipeline(&fixture_affine(), [64, 1, 1]).unwrap();
    let mut l = NativeList::default();
    l.set_compute_pipeline(&pso);
    l.set_compute_root_constants(0, &[1, 0]);
    l.set_compute_root_view(1, 0xDEAD_0000);
    l.set_compute_root_view(2, 0xDEAD_1000);
    l.dispatch(1, 1, 1).unwrap();
    let q = h.backend.create_queue().unwrap();
    let err = h.execute(&q, &[&l], &[], &[]).unwrap_err();
    assert!(err.contains("not in a mapped pool"), "{err}");
}

/// Pool buffers are made resident through one `MTLResidencySet`, added to each queue once, instead
/// of a `useResource` per dispatch; a buffer leaves it with its pool. The same work with sets off
/// names every buffer on every dispatch, and gives the same results.
#[test]
fn pool_buffers_are_resident_through_one_set_per_queue() {
    let Some(on) = host() else { return };
    if !on.backend.residency_sets() {
        eprintln!("skipped: no residency sets on this system");
        return;
    }
    let off = Host::new(Metal::new().unwrap().with_residency_sets(false));

    let work = |h: &Host, queues: usize, submissions: usize| {
        let pool = Pool::new(h, 4);
        let (src, dst) = (0, 16384);
        pool.write(src, &(0..N as u32).collect::<Vec<_>>());
        let pso = h.compute_pipeline(&fixture_affine(), [64, 1, 1]).unwrap();
        let qs: Vec<_> = (0..queues).map(|_| h.backend.create_queue().unwrap()).collect();
        let f = h.backend.create_fence(0).unwrap();
        let mut value = 0;
        for i in 0..submissions {
            let mut l = NativeList::default();
            l.set_compute_pipeline(&pso);
            l.set_compute_root_constants(0, &[i as u32 + 2, 1]);
            l.set_compute_root_view(1, pool.va(src));
            l.set_compute_root_view(2, pool.va(dst));
            l.dispatch((N / 64) as u32, 1, 1).unwrap();
            value += 1;
            h.execute(&qs[i % queues], &[&l], &[(&f, value - 1)], &[(&f, value)]).unwrap();
        }
        assert!(h.backend.wait(&f, value, 10_000));
        let last = submissions as u32 + 1;
        let out: Vec<u32> = pool.read(dst, N);
        assert!(out.iter().enumerate().all(|(i, v)| *v == i as u32 * last + 1), "{:?}", &out[..4]);
        pool
    };

    let _pool = work(&on, 2, 6);
    let c = on.backend.residency_counts();
    // One pool buffer, in the set once; the set is on both queues; no dispatch names a buffer.
    assert_eq!(c, victory_metal::metal::ResidencyCounts { use_resource: 0, added: 1, queues: 2 });

    let _pool = work(&off, 2, 6);
    let c = off.backend.residency_counts();
    assert_eq!((c.added, c.queues), (0, 0));
    assert!(c.use_resource >= 6 * 2, "{c:?}");
}

/// A pool that goes away takes its buffer out of the set (and does not break the next work).
#[test]
fn a_released_pool_leaves_the_residency_set() {
    let Some(h) = host() else { return };
    if !h.backend.residency_sets() {
        return;
    }
    let pool = Pool::new(&h, 2);
    pool.write(0, &(0..64u32).collect::<Vec<_>>());
    let pso = h.compute_pipeline(&fixture_affine(), [64, 1, 1]).unwrap();
    let dispatch = |pool: &Pool| {
        let mut l = NativeList::default();
        l.set_compute_pipeline(&pso);
        l.set_compute_root_constants(0, &[2, 0]);
        l.set_compute_root_view(1, pool.va(0));
        l.set_compute_root_view(2, pool.va(4096));
        l.dispatch(1, 1, 1).unwrap();
        run(&h, &[&l]);
    };
    dispatch(&pool);
    assert_eq!(h.backend.residency_counts().added, 1);
    let info = PoolInfo {
        id: 1,
        base: pool.mem.primary as usize,
        len: pool.mem.len,
        memory_entry: None,
    };
    h.memory().pool_released(&info);
    // The next resolve drops the released pool's buffer.
    let second = {
        let mem = Alias::new(2 * page_size().max(16384)).unwrap();
        let info = PoolInfo {
            id: 2,
            base: mem.primary as usize,
            len: mem.len,
            memory_entry: None,
        };
        h.memory().pool_created(&info);
        h.memory().mapped(mem.alias as usize, mem.len, &info, 0);
        Pool { mem }
    };
    second.write(0, &(0..64u32).collect::<Vec<_>>());
    dispatch(&second);
    assert_eq!(h.backend.residency_counts().added, 2);
    assert_eq!(second.read::<u32>(4096, 64), (0..64).map(|i| i * 2).collect::<Vec<_>>());
}

/// A draw left with no render target (the title's targets were of formats the host does not
/// represent) is dropped by itself: Metal opens no pass without an attachment, and the value
/// write recorded with it still runs.
#[test]
fn a_draw_with_no_render_target_is_dropped_alone() {
    let Some(h) = host() else { return };
    let pool = Pool::new(&h, 2);
    pool.write(
        0,
        &[
            vertex(-0.5, -0.5, [1.0, 0.0, 0.0, 1.0]),
            vertex(0.5, -0.5, [1.0, 0.0, 0.0, 1.0]),
            vertex(0.0, 0.5, [1.0, 0.0, 0.0, 1.0]),
        ],
    );
    let attrs = [
        VertexAttr {
            format: VertexFormat::Float2,
            slot: 0,
            offset: 0,
        },
        VertexAttr {
            format: VertexFormat::Float2,
            slot: 0,
            offset: 8,
        },
        VertexAttr {
            format: VertexFormat::Float4,
            slot: 0,
            offset: 16,
        },
    ];
    let pso = h
        .render_pipeline(&RenderPipelineDesc {
            vs: &fixture("tri_vs"),
            ps: Some(&fixture("tri_ps")),
            vs_layout: None,
            ps_layout: None,
            attributes: &attrs,
            streams: &[VertexStream {
                slot: 0,
                stride: 32,
                per_instance: false,
            }],
            color: &[Format::Rgba8Unorm],
            depth: None,
            depth_test: None,
            cull: Cull::None,
            ..Default::default()
        })
        .unwrap();
    let flag = 4096;
    pool.write(flag, &[0u64]);
    let mut l = NativeList::default();
    l.set_render_targets(&[], None);
    l.set_graphics_pipeline(&pso, Topology::Triangles);
    l.set_graphics_root_constants(0, &[0.5f32.to_bits(), 0, 0, 0]);
    l.set_vertex_buffer(0, pool.va(0));
    l.draw_instanced(3, 1, 0, 0).unwrap();
    l.write_value(pool.va(flag), 0xD6D, 8);
    run(&h, &[&l]);

    assert_eq!(h.no_targets(), 1);
    assert_eq!(pool.read::<u64>(flag, 1), [0xD6D], "the value write ran");
}

/// A dispatch that reads a root parameter nothing was set in is dropped by itself: the value
/// write recorded before it and the dispatch after it still run (a queue may wait for that value).
#[test]
fn a_dispatch_with_an_unset_root_parameter_is_dropped_alone() {
    let Some(h) = host() else { return };
    let pool = Pool::new(&h, 4);
    let (src, dst, flag) = (16384, 32768, 49152);
    pool.write(src, &(0..N as u32).collect::<Vec<_>>());
    pool.write(dst, &[0xEEEE_EEEEu32; N]);
    pool.write(flag, &[0u64]);
    let pso = h.compute_pipeline(&fixture_affine(), [64, 1, 1]).unwrap();

    let mut l = NativeList::default();
    l.write_value(pool.va(flag), 0xD6D, 8);
    l.set_compute_pipeline(&pso);
    l.set_compute_root_constants(0, &[5, 7]);
    // Root parameter 1 is not set yet: this dispatch cannot be given its arguments.
    l.set_compute_root_view(2, pool.va(dst));
    l.dispatch((N / 64) as u32, 1, 1).unwrap();
    l.set_compute_root_view(1, pool.va(src));
    l.dispatch((N / 64) as u32, 1, 1).unwrap();
    run(&h, &[&l]);

    assert_eq!(h.unset_roots(), 1);
    assert_eq!(pool.read::<u64>(flag, 1), [0xD6D], "the value write ran");
    let out: Vec<u32> = pool.read(dst, N);
    assert!(
        out.iter().enumerate().all(|(i, v)| *v == i as u32 * 5 + 7),
        "the second dispatch ran: {:?}",
        &out[..4]
    );
}

/// A descriptor entry can be repointed at the spare while its texture is gone,
/// then remirrored when the texture is realized again. This is the mechanism
/// that lets a heap-referenced texture be reclaimed without a dangling entry.
#[test]
fn an_evicted_descriptor_entry_is_repointed_to_the_spare_and_back() {
    let Some(h) = host() else { return };
    let white = h
        .texture(&TextureDesc::d2(Format::Rgba8Unorm, 1, 1))
        .unwrap();
    h.backend.upload_texture(&white, 0, 0, &[255; 4], 4, 0);
    h.register_texture(0xBEEF, &white);

    let views = vec![0u32; 4 * 4];
    let vstart = views.as_ptr() as u64;
    let vheap = h.heap(HeapKind::Resource, vstart, 4).unwrap();
    let mut srv = [0u32; 10];
    (srv[0], srv[1], srv[5]) = (28, 4, 1);
    crate::descriptor::create_srv(std::ptr::null_mut(), 0xBEEF as _, &srv, vstart);
    assert!(h.mirror(vstart).unwrap());
    assert_eq!(h.heap_texture_refs().get(&0xBEEF), Some(&1));
    assert_eq!(h.pending_textures(), 0);

    h.evict_texture_key(0xBEEF, Format::Rgba8Unorm);
    assert!(h.heap_texture_refs().is_empty(), "the real texture is released");
    assert_eq!(h.pending_textures(), 1, "the entry names the spare");

    h.register_texture(0xBEEF, &white);
    assert!(h.remirror_texture(0xBEEF), "the entry names the texture again");
    assert_eq!(h.heap_texture_refs().get(&0xBEEF), Some(&1));
    assert_eq!(h.pending_textures(), 0);

    h.release_texture_key(0xBEEF);
    assert!(h.heap_texture_refs().is_empty(), "release unpins the texture");
    assert_eq!(h.pending_textures(), 0, "a released view becomes null");

    h.drop_heap(vstart);
    drop(vheap);
    drop(views);
}

/// The encode-time pending path realizes a texture named by a copy command
/// before the submission; it is the same `realize_key` used for descriptor
/// entries, so a texture evicted between recording and encoding is restored.
#[test]
fn realize_pending_realizes_textures_a_copy_names() {
    let Some(h) = host() else { return };
    let pool = Pool::new(&h, 4);
    let res = crate::resource::test_texture_at(pool.va(0), (16, 16), 1, 28, Swizzle::LINEAR.0, 0);
    let key = res as usize;
    let mut list = NativeList::default();
    list.copy_texture(key, key);
    let lists = vec![Rc::new(list)];
    let mut stats = crate::native::title::Stats::default();
    with_title(|t| {
        let Title {
            host,
            pending,
            textures,
            backing,
            unloaded,
            ..
        } = t;
        assert!(!textures.contains_key(&key));
        crate::native::textures::realize_pending(
            host, pending, textures, backing, unloaded, &mut stats, 0, 0, &lists,
        )
        .unwrap();
        assert!(textures.contains_key(&key), "the copy's texture was realized");
    })
    .unwrap();
}

/// The queue's submitted/completed serials advance together after a fence
/// wait; descriptor repointing uses this gate to avoid rewriting a heap an
/// in-flight command buffer may still read.
#[test]
fn queue_progress_counts_committed_and_completed_buffers() {
    let Some(h) = host() else { return };
    let q = h.backend.create_queue().unwrap();
    let f = h.backend.create_fence(0).unwrap();
    let mut list = NativeList::default();
    let pool = Pool::new(&h, 1);
    list.fill(pool.va(0), 16, 0);
    h.execute(&q, &[&list], &[], &[(&f, 1)]).unwrap();
    assert!(h.backend.wait(&f, 1, 10_000));
    let (submitted, completed) = h.backend.queue_progress(&q);
    assert!(submitted >= 1, "the command buffer was counted");
    assert_eq!(completed, submitted, "the completion handler caught up");
}

/// A D3D12 null SRV (resource 0) is a defined null, not a texture awaiting
/// realization; it must not enter the pending set and abort strict mode.
#[test]
fn a_null_descriptor_is_not_a_pending_texture() {
    let Some(h) = host() else { return };
    let views = vec![0u32; 4 * 4];
    let vstart = views.as_ptr() as u64;
    let vheap = h.heap(HeapKind::Resource, vstart, 4).unwrap();
    let mut srv = [0u32; 10];
    (srv[0], srv[1], srv[5]) = (28, 4, 1);
    crate::descriptor::create_srv(std::ptr::null_mut(), std::ptr::null(), &srv, vstart);
    assert!(h.mirror(vstart).unwrap());
    assert_eq!(h.pending_textures(), 0, "a null view is not pending");
    assert!(h.heap_texture_refs().is_empty(), "a null view pins nothing");
    h.drop_heap(vstart);
    drop(vheap);
    drop(views);
}
