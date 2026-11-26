//! Title textures on the native host: textures laid out in the console's tiled layouts in title
//! memory, detiled into host textures; textures the title places in memory itself; render targets
//! cleared through the title's views. Our own synthetic texels. Skipped without Metal.

use super::Pool;
use crate::native::glue::{self, with_title};
use crate::native::memory::{PoolInfo, PoolObserver};
use crate::object::{This, slot};
use victory_backend::resources::*;
use victory_formats::texture::Swizzle;
use victory_metal::mem::{Alias, page_size};

fn pool(id: usize, pages: usize) -> Option<Pool> {
    // Needs a device; the title is made on first use.
    with_title(|_| ())?;
    let mem = Alias::new(pages * page_size().max(16384)).unwrap();
    let info = PoolInfo {
        id,
        base: mem.primary as usize,
        len: mem.len,
        memory_entry: None,
    };
    let observer = glue::pool_observer();
    observer.pool_created(&info);
    observer.mapped(mem.alias as usize, mem.len, &info, 0);
    Some(Pool { mem })
}

/// What `x`, `y` hold in mip `mip` of the test texture.
fn texel(mip: u32, x: u32, y: u32) -> [u8; 4] {
    [x as u8, y as u8, mip as u8 * 40, 255]
}

/// Tiles a texture's mips into `pool` memory at `at`, texel by `texel`.
fn tile(pool: &Pool, at: usize, layout: &victory_formats::texture::Layout) {
    let mut bytes = vec![0u8; layout.size as usize];
    for mip in 0..layout.mips.len() as u32 {
        let m = &layout.mips[mip as usize];
        let linear: Vec<u8> = (0..m.height)
            .flat_map(|y| (0..m.width).flat_map(move |x| texel(mip, x, y)))
            .collect();
        layout
            .tile(&mut bytes, mip, 0, &linear, m.width as usize * 4, linear.len())
            .unwrap();
    }
    pool.write(at, &bytes);
}

fn read_mip(key: usize, mip: usize, w: usize, h: usize) -> Vec<u8> {
    with_title(|t| {
        let tex = t.texture_of(key).expect("a host texture");
        let mut out = vec![0u8; w * h * 4];
        t.host.backend.read_texture(&tex, mip, 0, &mut out, w * 4, 0);
        out
    })
    .unwrap()
}

/// A tiled texture with a mip chain is detiled mip by mip into its host texture when work is
/// submitted, again only when its memory changed.
#[test]
fn a_tiled_title_texture_is_detiled_into_the_host_texture() {
    let Some(pool) = pool(9101, 8) else { return };
    // The XOR modes (`_X`, `_T`) too: their equations come from the pipe configuration, not a
    // fixed pattern, and a texture of theirs is no longer left empty.
    for swizzle in [
        Swizzle::LINEAR,
        Swizzle::SW_256B_S,
        Swizzle::SW_4KB_S,
        Swizzle::SW_64KB_D,
        Swizzle::SW_4KB_S_X,
        Swizzle::SW_64KB_D_X,
        Swizzle::SW_64KB_S_T,
    ] {
        let at = 0;
        let res = crate::resource::test_texture_at(pool.va(at), (64, 32), 4, 28, swizzle.0, 0);
        let info = crate::resource::texture_info(res).unwrap();
        tile(&pool, at, &info.layout);
        let key = res as usize;
        let uploads = |key| with_title(|t| t.uploads_of(key)).unwrap();
        with_title(|t| t.sync_textures());
        // Nothing named the texture yet: no host texture, nothing detiled.
        assert_eq!(uploads(key), 0);
        let mips: Vec<(usize, usize)> = (0..4).map(|m| (64 >> m, 32 >> m)).collect();
        // Naming it makes it; the next sync fills it.
        with_title(|t| {
            assert!(t.realize(key));
            t.sync_textures();
        });
        for (mip, &(w, h)) in mips.iter().enumerate() {
            let out = read_mip(key, mip, w, h);
            for (x, y) in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1), (w / 2, h / 2)] {
                let at = (y * w + x) * 4;
                assert_eq!(
                    out[at..at + 4],
                    texel(mip as u32, x as u32, y as u32),
                    "{swizzle:?} mip {mip} at {x},{y}"
                );
            }
        }
        assert_eq!(uploads(key), 1, "uploaded once");
        // Unchanged memory is not detiled again; changed memory is.
        with_title(|t| t.sync_textures());
        assert_eq!(uploads(key), 1);
        pool.write(at, &[0xFFu8; 16]);
        with_title(|t| t.sync_textures());
        assert_eq!(uploads(key), 2);
    }
}

/// A texture uploaded blank and written whole later (a movie's planes) is uploaded again at the
/// next submission, not a scanner round later (a round is seconds long once the title streams).
#[test]
fn a_texture_uploaded_blank_is_probed_at_each_submission() {
    // 256 KiB: 64 pages, more than a sampling probe reads.
    let Some(pool) = pool(9105, (256 << 10) / page_size() + 1) else { return };
    let res = crate::resource::test_texture_at(pool.va(0), (256, 256), 1, 28, Swizzle::LINEAR.0, 0);
    let info = crate::resource::texture_info(res).unwrap();
    let key = res as usize;
    with_title(|t| {
        assert!(t.realize(key));
        t.sync_dirty();
        assert_eq!(t.uploads_of(key), 1, "the blank first upload");
    });
    pool.write(0, &vec![0xABu8; info.size]);
    with_title(|t| {
        t.sync_dirty();
        assert_eq!(t.uploads_of(key), 2, "uploaded again at the next submission");
    });
    // A glyph's worth of bytes in page 3 (a page a sampling probe would skip).
    pool.write(3 * 4096 + 100, &[1u8; 16]);
    with_title(|t| {
        t.sync_dirty();
        assert_eq!(t.uploads_of(key), 3, "a small write is seen at the next submission too");
    });
}

/// Two title resource objects over the same title-memory range and layout are aliases: they share
/// one host texture, and the second costs no additional device allocation.
#[test]
fn identical_title_resources_share_one_host_texture() {
    let Some(pool) = pool(9109, 8) else { return };
    let a = crate::resource::test_texture_at(pool.va(0), (32, 32), 1, 28, Swizzle::LINEAR.0, 0);
    let b = crate::resource::test_texture_at(pool.va(0), (32, 32), 1, 28, Swizzle::LINEAR.0, 0);
    let (same, allocated, keys, unique) = with_title(|t| {
        assert!(t.realize(a as usize));
        let allocated = t.allocated();
        assert!(t.realize(b as usize));
        let pa = std::rc::Rc::as_ptr(&t.texture_of(a as usize).unwrap());
        let pb = std::rc::Rc::as_ptr(&t.texture_of(b as usize).unwrap());
        (pa == pb, allocated == t.allocated(), t.key_bytes(), t.resident().1)
    })
    .unwrap();
    assert!(same, "aliases share the backing");
    assert!(allocated, "the second alias allocates nothing");
    let info = crate::resource::texture_info(a).unwrap();
    assert!(keys >= 2 * info.size, "both keys count toward title bytes");
    assert!(unique >= info.size, "one backing counts once");
}

/// A block-compressed texture: 16 bytes a 4x4 block, in the console's layout, detiled as blocks.
#[test]
fn a_block_compressed_title_texture_is_detiled_as_blocks() {
    let Some(pool) = pool(9102, 4) else { return };
    // BC7 (98): a 32x16 texture is 8x4 blocks.
    let res = crate::resource::test_texture_at(pool.va(0), (32, 16), 1, 98, Swizzle::SW_64KB_D.0, 0);
    let info = crate::resource::texture_info(res).unwrap();
    assert_eq!(info.layout.desc.element_bytes, 16);
    let m = &info.layout.mips[0];
    let blocks: Vec<u8> = (0..m.width * m.height * 16).map(|i| (i * 7) as u8).collect();
    let mut bytes = vec![0u8; info.layout.size as usize];
    info.layout
        .tile(&mut bytes, 0, 0, &blocks, m.width as usize * 16, blocks.len())
        .unwrap();
    pool.write(0, &bytes);
    let key = res as usize;
    with_title(|t| {
        assert!(t.realize(key));
        t.sync_textures();
        let tex = t.texture_of(key).unwrap();
        let mut out = vec![0u8; blocks.len()];
        t.host.backend.read_texture(&tex, 0, 0, &mut out, m.width as usize * 16, 0);
        assert_eq!(out, blocks);
    });
}

/// A texture a view names by memory address and description, without a resource object (the
/// console's `CreateShaderResourceViewPlaced`): the host makes it when the view is written.
#[test]
fn a_placed_texture_view_makes_and_fills_its_texture() {
    let Some(pool) = pool(9103, 4) else { return };
    let res = crate::resource::test_texture_at(pool.va(0), (32, 32), 1, 28, Swizzle::SW_64KB_S.0, 0);
    let info = crate::resource::texture_info(res).unwrap();
    tile(&pool, 0, &info.layout);
    let words = crate::resource::texture_desc(res).unwrap();
    let heap = crate::descriptor::test_heap(false, 4);
    let start = slot::<extern "system" fn(This) -> u64>(heap, 0x48)(heap);
    let mut view = [0u32; 16];
    (view[0], view[1], view[5]) = (28, 4, 1);
    (view[10], view[11]) = (pool.va(0) as u32, (pool.va(0) >> 32) as u32);
    crate::descriptor::create_placed_srv(std::ptr::null(), &words, &view, start);
    // The heap is made (and the view written into it) when a list names it.
    let key = crate::native::textures::placed_key(pool.va(0), &words);
    with_title(|t| {
        assert!(
            !t.host.knows(key),
            "the descriptor is pending until a list can index it"
        );
        assert!(t.realize(key), "realization makes the placed texture");
        t.sync_textures();
    });
    let out = read_mip(key, 0, 32, 32);
    assert_eq!(out[(5 * 32 + 9) * 4..][..4], texel(0, 9, 5));
}

/// A render target the title clears through its view: the host texture is the target, never
/// filled from memory.
#[test]
fn a_render_target_is_cleared_through_the_titles_view() {
    let Some(pool) = pool(9104, 4) else { return };
    let rt = crate::resource::test_texture_at(pool.va(0), (16, 16), 1, 28, Swizzle::SW_64KB_R_X.0, 1);
    pool.write(0, &[0x7Fu8; 64]);
    // An RTV in descriptor memory the title owns.
    let heap = crate::descriptor::test_heap(false, 4);
    let handle = slot::<extern "system" fn(This) -> u64>(heap, 0x48)(heap);
    crate::descriptor::create_rtv(std::ptr::null(), rt, std::ptr::null(), handle);
    let list = crate::list::test_list(false, std::ptr::null());
    let (q, f) = (crate::queue::test_queue(false), {
        let mut f = std::ptr::null();
        crate::fence::create(std::ptr::null(), 0, 0, &crate::fence::FENCE_IIDS[0], &mut f);
        f
    });
    // OMSetRenderTargets(1, &handle, false, null), then ClearRenderTargetView(handle, color).
    slot::<extern "system" fn(This, u32, *const u64, i32, *const u64)>(list, 0x190)(
        list,
        1,
        &handle,
        0,
        std::ptr::null(),
    );
    let color = [0.0f32, 1.0, 0.0, 1.0];
    slot::<extern "system" fn(This, u64, *const [f32; 4], u32, *const u8)>(list, 0x1a0)(
        list,
        handle,
        &color,
        0,
        std::ptr::null(),
    );
    assert_eq!(slot::<extern "system" fn(This) -> i32>(list, 0x48)(list), 0);
    slot::<extern "system" fn(This, u32, *const This)>(q, 0x50)(q, 1, &list);
    slot::<extern "system" fn(This, This, u64) -> i32>(q, 0x70)(q, f, 1);
    assert_eq!(
        slot::<extern "system" fn(This, u64, *mut std::ffi::c_void) -> i32>(f, 0x48)(
            f,
            1,
            std::ptr::null_mut()
        ),
        0
    );
    let out = read_mip(rt as usize, 0, 16, 16);
    assert!(out.chunks(4).all(|p| p == [0, 255, 0, 255]), "{:?}", &out[..8]);
}

/// A layout the host cannot address (`SW_4KB_R_X`, which has no equations) is counted and left empty, not
/// an error that stops the submission.
#[test]
fn an_undetilable_layout_is_counted_not_fatal() {
    let Some(pool) = pool(9105, 4) else { return };
    let res = crate::resource::test_texture_at(pool.va(0), (64, 64), 1, 28, Swizzle::SW_4KB_R_X.0, 0);
    let before = with_title(|t| t.stats.undetilable).unwrap();
    with_title(|t| {
        assert!(t.realize(res as usize));
        t.sync_textures();
        t.sync_textures();
    });
    assert_eq!(with_title(|t| t.stats.undetilable).unwrap(), before + 1, "once per content");
}

/// `PresentX`: the first resource of the largest plane is drawn over the app's layer (scaled,
/// whatever its channel order), after the queue's earlier work. Offscreen: the layer is not in a
/// window; its last drawable is read back.
#[test]
fn present_draws_the_back_buffer_over_the_apps_layer() {
    let Some(pool) = pool(9106, 4) else { return };
    let rt = crate::resource::test_texture_at(pool.va(0), (16, 16), 1, 28, Swizzle::SW_64KB_S.0, 1);
    let heap = crate::descriptor::test_heap(false, 4);
    let handle = slot::<extern "system" fn(This) -> u64>(heap, 0x48)(heap);
    crate::descriptor::create_rtv(std::ptr::null(), rt, std::ptr::null(), handle);
    let layer = glue::create_layer(32, 32).expect("a layer");
    glue::set_layer(layer.as_ptr()).expect("a surface");
    let (q, f) = (crate::queue::test_queue(false), {
        let mut f = std::ptr::null();
        crate::fence::create(std::ptr::null(), 0, 0, &crate::fence::FENCE_IIDS[0], &mut f);
        f
    });
    let list = crate::list::test_list(false, std::ptr::null());
    slot::<extern "system" fn(This, u32, *const u64, i32, *const u64)>(list, 0x190)(
        list,
        1,
        &handle,
        0,
        std::ptr::null(),
    );
    // Red in the title's RGBA.
    let color = [1.0f32, 0.0, 0.0, 1.0];
    slot::<extern "system" fn(This, u64, *const [f32; 4], u32, *const u8)>(list, 0x1a0)(
        list,
        handle,
        &color,
        0,
        std::ptr::null(),
    );
    slot::<extern "system" fn(This) -> i32>(list, 0x48)(list);
    slot::<extern "system" fn(This, u32, *const This)>(q, 0x50)(q, 1, &list);
    // One plane: the frame token, one resource, its address.
    #[repr(C)]
    struct Plane {
        token: u64,
        count: u32,
        pad: u32,
        resources: *const This,
    }
    let plane = Plane {
        token: 1,
        count: 1,
        pad: 0,
        resources: &rt,
    };
    let before = glue::frames_presented();
    let present = slot::<extern "system" fn(This, u32, *const Plane, *const u8) -> i32>(q, 0xf0);
    assert_eq!(present(q, 1, &plane, std::ptr::null()), 0);
    assert_eq!(glue::frames_presented(), before + 1);
    // A signal after the present: the frame is drawn when the fence is reached.
    slot::<extern "system" fn(This, This, u64) -> i32>(q, 0x70)(q, f, 1);
    assert_eq!(
        slot::<extern "system" fn(This, u64, *mut std::ffi::c_void) -> i32>(f, 0x48)(f, 1, std::ptr::null_mut()),
        0
    );
    let (w, h, bgra) = glue::last_frame().expect("a frame");
    assert_eq!((w, h), (32, 32));
    assert!(
        bgra.chunks(4).all(|p| p == [0, 0, 255, 255]),
        "red as BGRA: {:?}",
        &bgra[..8]
    );
    drop(layer);
}

/// A depth/stencil console texture has a 4-byte depth plane. Metal's
/// `replaceRegion` on D32FloatS8 takes that depth-plane row (4 bytes/texel),
/// not an expanded 8-byte host row; the stencil plane is not uploaded.
#[test]
fn a_depth_stencil_texture_uploads_the_depth_plane() {
    let Some(pool) = pool(9107, 4) else { return };
    let res = crate::resource::test_texture_at(pool.va(0), (16, 16), 1, 19, Swizzle::LINEAR.0, 0);
    let info = crate::resource::texture_info(res).unwrap();
    assert_eq!(info.layout.desc.element_bytes, 4, "console depth plane is four bytes");
    assert_eq!(info.desc().unwrap().format, Format::D32FloatS8);
    tile(&pool, 0, &info.layout);
    let key = res as usize;
    let out = with_title(|t| {
        assert!(t.realize(key));
        t.sync_textures();
        let tex = t.texture_of(key).unwrap();
        let mut out = vec![0u8; 16 * 16 * 4];
        t.host
            .backend
            .read_texture(&tex, 0, 0, &mut out, 16 * 4, 16 * 16 * 4);
        out
    })
    .unwrap();
    for (x, y) in [(0, 0), (9, 5), (15, 15)] {
        let at = (y * 16 + x) * 4;
        assert_eq!(&out[at..at + 4], &texel(0, x as u32, y as u32), "depth at {x},{y}");
    }
}
