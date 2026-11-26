//! Render passes on Metal: a vertex and a pixel shader (our own HLSL, dxc, Metal Shader Converter)
//! draw offscreen with an input layout read from zero-copy vertex memory, a descriptor-table texture
//! and sampler in the pixel shader, a colour and a depth attachment; the pixels are read back.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use victory_metal::backend::{ComputeBackend, Dispatch, Heaps, RootArgs};
use victory_metal::metal::Metal;
use victory_metal::resources::*;

const W: usize = 16;

/// Everything one test draws with.
struct Scene<'a> {
    m: &'a Metal,
    pipe: <Metal as GraphicsBackend>::RenderPipeline,
    arena: Arena,
    res: <Metal as ComputeBackend>::Heap,
    smp: <Metal as ComputeBackend>::Heap,
    color: <Metal as ResourceBackend>::Texture,
    depth: <Metal as ResourceBackend>::Texture,
}

type Vertex = [f32; 8];

fn v(x: f32, y: f32, c: [f32; 4]) -> Vertex {
    [x, y, 0.5, 0.5, c[0], c[1], c[2], c[3]]
}

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];

impl<'a> Scene<'a> {
    fn new(m: &'a Metal, depth_test: Option<(bool, Compare)>) -> Scene<'a> {
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
        let pipe = m
            .create_render_pipeline(&RenderPipelineDesc {
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
                depth: Some(Format::D32Float),
                depth_test,
                cull: Cull::None,
                ..Default::default()
            })
            .unwrap();
        let mut rt = TextureDesc::d2(Format::Rgba8Unorm, W, W);
        rt.render_target = true;
        let color = m.create_texture(&rt).unwrap();
        rt.format = Format::D32Float;
        let depth = m.create_texture(&rt).unwrap();
        // The pixel shader multiplies by a 1x1 white texture sampled through the heaps.
        let white = m
            .create_texture(&TextureDesc::d2(Format::Rgba8Unorm, 1, 1))
            .unwrap();
        m.upload_texture(&white, 0, 0, &[255; 4], 4, 0);
        let s = m
            .create_sampler(&SamplerDesc {
                min_mag: Filter::Nearest,
                mip: Filter::Nearest,
                address: [Address::Clamp; 3],
                max_anisotropy: 1,
            })
            .unwrap();
        let res = m.create_heap(HeapKind::Resource, 8).unwrap();
        let smp = m.create_heap(HeapKind::Sampler, 2).unwrap();
        m.write_texture(&res, 0, &white, 0.0);
        m.write_sampler(&smp, 0, &s, 0.0);
        Scene {
            m,
            pipe,
            arena: Arena::new(m, 2),
            res,
            smp,
            color,
            depth,
        }
    }

    fn root(&self, z: f32) -> RootArgs {
        RootArgs::default()
            .constants(&[z.to_bits(), 0, 0, 0])
            .address(self.m.heap_address(&self.res, 0))
            .address(self.m.heap_address(&self.smp, 0))
    }

    fn draw(&self, z: f32, call: DrawCall, verts: usize, idx: bool) -> Draw<'_, Metal> {
        Draw {
            pipeline: &self.pipe,
            topology: Topology::Triangles,
            state: DrawState::default(),
            root: self.root(z),
            fragment_root: None,
            heaps: Heaps {
                resource: Some(&self.res),
                sampler: Some(&self.smp),
            },
            resources: vec![],
            vertex_buffers: vec![(&self.arena.buf, self.arena.at(verts))],
            index_buffer: idx.then(|| (&self.arena.buf, self.arena.at(4096) as *const u8)),
            call,
        }
    }

    fn pass<'b>(&'b self, draws: Vec<Draw<'b, Metal>>, clear: [f32; 4]) -> Pass<'b, Metal> {
        Pass::Render(RenderPass {
            color: vec![ColorTarget {
                texture: &self.color,
                clear: Some(clear),
            }],
            depth: Some(DepthTarget {
                texture: &self.depth,
                clear: Some(1.0),
            }),
            draws,
        })
    }

    fn submit(&self, passes: &[Pass<'_, Metal>]) {
        let q = self.m.create_queue().unwrap();
        let f = self.m.create_fence(0).unwrap();
        self.m
            .submit_passes(
                &q,
                &PassSubmission {
                    waits: &[],
                    passes,
                    signals: &[(&f, 1)],
                },
            )
            .unwrap();
        assert!(self.m.wait(&f, 1, 10_000), "GPU did not finish");
    }

    fn pixels(&self) -> Vec<[u8; 4]> {
        let mut out = vec![0u8; W * W * 4];
        self.m.read_texture(&self.color, 0, 0, &mut out, W * 4, 0);
        out.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()
    }
}

fn px(p: &[[u8; 4]], x: usize, y: usize) -> [u8; 4] {
    p[y * W + x]
}

#[test]
fn triangle_offscreen() {
    let Some(m) = backend() else { return };
    let s = Scene::new(&m, Some((true, Compare::Less)));
    s.arena.write(
        0,
        &[v(-0.5, -0.5, RED), v(0.5, -0.5, RED), v(0.0, 0.5, RED)],
    );
    let d = s.draw(
        0.5,
        DrawCall::Arrays {
            vertex_count: 3,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        },
        0,
        false,
    );
    s.submit(&[s.pass(vec![d], [0.0, 0.0, 1.0, 1.0])]);
    let p = s.pixels();
    assert_eq!(px(&p, 8, 8), [255, 0, 0, 255], "inside the triangle");
    assert_eq!(px(&p, 0, 0), [0, 0, 255, 255], "cleared corner");
    assert_eq!(px(&p, 15, 15), [0, 0, 255, 255], "cleared corner");
    // Direct 'up' check: the apex is at y = +0.5, which is the upper rows of the image (row 4),
    // the base at y = -0.5 the lower rows (row 12): wide at the bottom, narrow at the top.
    let width = |y| (0..W).filter(|x| px(&p, *x, y) == [255, 0, 0, 255]).count();
    assert!(
        width(11) > width(5) && width(5) > 0,
        "{} {}",
        width(11),
        width(5)
    );
}

#[test]
fn texture_copy_and_region_offscreen() {
    let Some(m) = backend() else { return };
    let desc = TextureDesc::d2(Format::Rgba8Unorm, 4, 4);
    let src = m.create_texture(&desc).unwrap();
    let dst = m.create_texture(&desc).unwrap();
    let pattern: Vec<u8> = (0..4 * 4 * 4).map(|i| (i as u8).wrapping_mul(7)).collect();
    m.upload_texture(&src, 0, 0, &pattern, 16, 64);
    m.upload_texture(&dst, 0, 0, &[0u8; 4 * 4 * 4], 16, 64);

    let q = m.create_queue().unwrap();
    let f = m.create_fence(0).unwrap();
    m.submit_passes(
        &q,
        &PassSubmission {
            waits: &[],
            passes: &[Pass::Blit(vec![BlitOp::CopyTexture {
                src: &src,
                dst: &dst,
            }])],
            signals: &[(&f, 1)],
        },
    )
    .unwrap();
    assert!(m.wait(&f, 1, 10_000), "copy did not finish");
    let mut out = vec![0u8; 4 * 4 * 4];
    m.read_texture(&dst, 0, 0, &mut out, 16, 64);
    assert_eq!(out, pattern, "whole-resource copy");

    m.upload_texture(&dst, 0, 0, &[0u8; 4 * 4 * 4], 16, 64);
    m.submit_passes(
        &q,
        &PassSubmission {
            waits: &[],
            passes: &[Pass::Blit(vec![BlitOp::CopyTextureRegion {
                src: &src,
                dst: &dst,
                src_slice: 0,
                src_level: 0,
                src_origin: [1, 1, 0],
                size: [2, 2, 1],
                dst_slice: 0,
                dst_level: 0,
                dst_origin: [0, 0, 0],
            }])],
            signals: &[(&f, 2)],
        },
    )
    .unwrap();
    assert!(m.wait(&f, 2, 10_000), "region copy did not finish");
    m.read_texture(&dst, 0, 0, &mut out, 16, 64);
    let at = |x: usize, y: usize| &out[(y * 4 + x) * 4..][..4];
    let src_at = |x: usize, y: usize| &pattern[(y * 4 + x) * 4..][..4];
    assert_eq!(at(0, 0), src_at(1, 1), "region origin");
    assert_eq!(at(1, 1), src_at(2, 2), "region extent");
    assert_eq!(at(2, 2), [0, 0, 0, 0], "outside the region stays clear");
}

#[test]
fn indexed_draw_with_depth_test_and_depth_sampling() {
    let Some(m) = backend() else { return };
    let s = Scene::new(&m, Some((true, Compare::Less)));
    // Left half: an indexed quad (4 vertices, 6 u16 indices) at z 0.25, green. Drawn second, a
    // full-screen red triangle at z 0.75 is behind it where they overlap and in front of nothing else.
    s.arena.write(
        0,
        &[
            v(-1.0, -1.0, GREEN),
            v(0.0, -1.0, GREEN),
            v(0.0, 1.0, GREEN),
            v(-1.0, 1.0, GREEN),
        ],
    );
    s.arena.write(4096, &[0u16, 1, 2, 0, 2, 3]);
    s.arena.write(
        512,
        &[v(-1.0, -1.0, RED), v(3.0, -1.0, RED), v(-1.0, 3.0, RED)],
    );
    let quad = s.draw(
        0.25,
        DrawCall::Indexed {
            index_count: 6,
            instance_count: 1,
            first_index: 0,
            base_vertex: 0,
            first_instance: 0,
            wide: false,
        },
        0,
        true,
    );
    let tri = s.draw(
        0.75,
        DrawCall::Arrays {
            vertex_count: 3,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        },
        512,
        false,
    );

    // After the render pass, a compute pass samples the depth attachment itself (Metal cannot view
    // Depth32Float as R32Float: "not castable"), as `Texture2D<float4>`; depth is in .x.
    let cells = W * W;
    let out = Arena::new(&m, 1);
    let p = m.create_compute_pipeline(&fixture("sample2d")).unwrap();
    let res = m.create_heap(HeapKind::Resource, 2).unwrap();
    m.write_texture(&res, 0, &s.depth, 0.0);
    m.write_buffer(&res, 1, &out.buf, out.at(0), (cells * 16) as u32);
    let sm = m
        .create_sampler(&SamplerDesc {
            min_mag: Filter::Nearest,
            mip: Filter::Nearest,
            address: [Address::Clamp; 3],
            max_anisotropy: 1,
        })
        .unwrap();
    m.write_sampler(&s.smp, 1, &sm, 0.0);
    let sample = [Dispatch {
        pipeline: &p,
        root: RootArgs::default()
            .constants(&[W as u32, W as u32])
            .address(m.heap_address(&res, 0))
            .address(m.heap_address(&s.smp, 1)),
        resources: vec![],
        heaps: Heaps {
            resource: Some(&res),
            sampler: Some(&s.smp),
        },
        groups: [W / 4, W / 4, 1],
        threads_per_group: [4, 4, 1],
    }];
    s.submit(&[
        s.pass(vec![quad, tri], [0.0, 0.0, 0.0, 1.0]),
        Pass::Compute(&sample),
    ]);

    let p = s.pixels();
    assert_eq!(px(&p, 4, 8), [0, 255, 0, 255], "quad (z 0.25) in front");
    assert_eq!(
        px(&p, 12, 8),
        [255, 0, 0, 255],
        "triangle where only it covers"
    );
    let depth: Vec<f32> = out.read(0, cells * 4);
    let d = |x: usize, y: usize| depth[(y * W + x) * 4];
    assert!((d(4, 8) - 0.25).abs() < 1e-6, "depth left {}", d(4, 8));
    assert!((d(12, 8) - 0.75).abs() < 1e-6, "depth right {}", d(12, 8));
}

#[test]
fn depth_test_rejects_a_draw_behind() {
    let Some(m) = backend() else { return };
    let s = Scene::new(&m, Some((true, Compare::Less)));
    // The same full-screen triangle twice: green at z 0.25 first, red at z 0.75 second. Red fails.
    s.arena.write(
        0,
        &[
            v(-1.0, -1.0, GREEN),
            v(3.0, -1.0, GREEN),
            v(-1.0, 3.0, GREEN),
        ],
    );
    s.arena.write(
        512,
        &[v(-1.0, -1.0, RED), v(3.0, -1.0, RED), v(-1.0, 3.0, RED)],
    );
    let call = DrawCall::Arrays {
        vertex_count: 3,
        instance_count: 1,
        first_vertex: 0,
        first_instance: 0,
    };
    let (a, b) = (s.draw(0.25, call, 0, false), s.draw(0.75, call, 512, false));
    s.submit(&[s.pass(vec![a, b], [0.0, 0.0, 0.0, 1.0])]);
    assert!(s.pixels().iter().all(|p| *p == [0, 255, 0, 255]));
}
