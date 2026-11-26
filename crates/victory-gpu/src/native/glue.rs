//! The one native host the title-facing objects call on macOS: `queue.rs`, `fence.rs`, `list.rs`,
//! `descriptor.rs` and `resource.rs` forward to the functions here (the same call sites that go to
//! `crate::host` on Windows).
//!
//! Threads. The title calls from many threads; the backend's objects and the `Rc`s inside
//! [`Title`] are not thread safe, so everything that touches them runs under one lock (`TITLE`),
//! always taken before a list's own lock, and objects holding such `Rc`s (lists) are dropped under
//! it. The GPU never waits on the lock: submissions are asynchronous, and a timer thread moves held
//! work and fence values along.

use super::memory::{PoolObserver, PoolTable};
use super::state::Graphics;
use super::title::{Guessed, Pipe, TargetFormats, Title, TitleList, Wait};
use crate::object::This;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use victory_backend::resources::HeapKind;
use victory_metal::metal::Metal;

pub type MetalTitle = Title<Metal>;
pub type MetalList = TitleList<Metal>;

struct Shared(MetalTitle);
// SAFETY: every access to the title's `Rc`s and Objective-C objects is under `TITLE`.
unsafe impl Send for Shared {}

static TITLE: Mutex<Option<Shared>> = Mutex::new(None);
static FAILED: AtomicBool = AtomicBool::new(false);
static POOLS: OnceLock<Arc<PoolTable>> = OnceLock::new();

/// The table of the title's memory pools. Feed it through [`pool_observer`].
pub fn pools() -> &'static Arc<PoolTable> {
    POOLS.get_or_init(Default::default)
}

/// What `victory_hle`'s physical-memory observer forwards its four calls to
/// (`pool_created`, `pool_released`, `mapped`, `unmapped`): a plain handle on the pool table the
/// native host reads, safe to call from any title thread, before or after the GPU exists.
pub fn pool_observer() -> Arc<dyn PoolObserver> {
    pools().clone()
}

/// Called with the event handle (as the title gave it) when a fence a title waits on with
/// `SetEventOnCompletion` is reached. The runtime sets it; without one, events are not signalled.
static EVENT_HOOK: OnceLock<fn(usize)> = OnceLock::new();

pub fn set_event_hook(hook: fn(usize)) {
    let _ = EVENT_HOOK.set(hook);
}

pub(crate) fn signal_event(event: usize) {
    if let Some(hook) = EVENT_HOOK.get() {
        hook(event);
    }
}

fn lock() -> std::sync::MutexGuard<'static, Option<Shared>> {
    TITLE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Runs `f` on the native host, made on first use; `None` if there is no Metal device (the call
/// is then ignored, as when there is no host GPU on Windows).
pub(crate) fn with_title<R>(f: impl FnOnce(&mut MetalTitle) -> R) -> Option<R> {
    let mut guard = lock();
    if guard.is_none() {
        if FAILED.load(Ordering::Relaxed) {
            return None;
        }
        match Metal::new() {
            Ok(metal) => {
                tracing::info!("native host: Metal");
                *guard = Some(Shared(Title::new(metal, pools().clone())));
                start_timer();
            }
            Err(e) => {
                FAILED.store(true, Ordering::Relaxed);
                tracing::warn!(error = %e, "no native host GPU");
                return None;
            }
        }
    }
    guard.as_mut().map(|t| f(&mut t.0))
}

/// Moves held work along and raises the title's copies of fence values as the GPU signals them
/// (titles read a fence's value from memory), without a call of the title's to drive it.
fn start_timer() {
    let _ = std::thread::Builder::new()
        .name("victory native host".into())
        .spawn(|| {
            loop {
                std::thread::sleep(std::time::Duration::from_millis(1));
                if let Some(t) = lock().as_mut() {
                    refresh(&mut t.0);
                }
            }
        });
}

fn refresh(t: &mut MetalTitle) {
    t.pump();
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if last.is_none_or(|l| l.elapsed().as_secs() >= 5) {
            let stuck = t.stuck(std::time::Duration::from_secs(2));
            if !stuck.is_empty() {
                *last = Some(std::time::Instant::now());
                tracing::warn!("work held for a wait that does not hold: {stuck:?}");
            }
        }
    }
    for (fence, value) in t.completed() {
        crate::fence::raise(fence as This, value);
    }
}

/// Runs `f` on the native host and a title list's recording.
pub(crate) fn with_list<R>(
    this: This,
    f: impl FnOnce(&mut MetalTitle, &mut MetalList) -> R,
) -> Option<R> {
    if this.is_null() {
        return None;
    }
    with_title(|t| {
        let list = crate::list::native(this);
        let mut list = list.lock().unwrap_or_else(|e| e.into_inner());
        f(t, &mut list)
    })
}

// --- queues and fences ---

/// `ExecuteCommandLists`.
pub(crate) fn submit(queue: This, lists: &[This]) {
    with_title(|t| {
        if t.ensure_queue(queue as usize).is_err() {
            return;
        }
        let mut segments = Vec::new();
        for &l in lists.iter().filter(|l| !l.is_null()) {
            let list = crate::list::native(l);
            segments.extend(list.lock().unwrap_or_else(|e| e.into_inner()).segments());
        }
        t.submit(queue as usize, segments);
    });
}

pub(crate) fn queue_signal(queue: This, fence: This, value: u64) {
    let initial = crate::fence::initial(fence);
    with_title(|t| {
        let _ = t.ensure_queue(queue as usize);
        t.signal_queue(queue as usize, fence as usize, initial, value);
    });
}

pub(crate) fn queue_wait(queue: This, fence: This, value: u64) {
    let initial = crate::fence::initial(fence);
    with_title(|t| {
        let _ = t.ensure_queue(queue as usize);
        t.wait_queue(queue as usize, fence as usize, initial, value);
    });
}

pub(crate) fn queue_released(queue: This) {
    with_title(|t| t.forget_queue(queue as usize));
}

/// `ID3D12Fence::Signal`.
pub(crate) fn fence_signal(fence: This, value: u64) {
    let initial = crate::fence::initial(fence);
    with_title(|t| t.signal_cpu(fence as usize, initial, value));
}

/// The value fence `fence` has completed, queues pumped first.
pub(crate) fn fence_poll(fence: This) -> u64 {
    let initial = crate::fence::initial(fence);
    with_title(|t| {
        t.pump();
        t.fence_value(fence as usize, initial)
    })
    .unwrap_or(initial)
}

/// `GetCompletedValue` of a fence that has no GPU behind it yet: nothing to ask.
pub(crate) fn fence_released(fence: This) {
    with_title(|t| t.forget_fence(fence as usize));
}

/// Blocks until fence `fence` reaches `value` (the title's `SetEventOnCompletion` with no event):
/// held work keeps moving while it waits. False after `timeout`.
pub(crate) fn fence_wait(fence: This, value: u64, timeout: std::time::Duration) -> bool {
    let initial = crate::fence::initial(fence);
    let start = std::time::Instant::now();
    loop {
        match with_title(|t| t.try_wait(fence as usize, initial, value)) {
            Some(true) => return true,
            None => return false,
            Some(false) => {}
        }
        if start.elapsed() >= timeout {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_micros(200));
    }
}

// --- present ---

/// Gives the host the app's drawable layer (a `CAMetalLayer`, by address; it must outlive the
/// host): frames the title presents are drawn into it from then on. Without one, `PresentX` shows
/// nothing.
pub fn set_layer(layer: *mut std::ffi::c_void) -> Result<(), String> {
    with_title(|t| t.set_surface(layer)).unwrap_or_else(|| Err("no native host GPU".into()))
}

/// A layer drawing to the host's device, `width` by `height`, for an app that has none (or a test):
/// hand [`victory_metal::present::Layer::as_ptr`] to [`set_layer`] and keep the layer alive.
pub fn create_layer(width: usize, height: usize) -> Result<victory_metal::present::Layer, String> {
    with_title(|t| t.host.backend.create_layer(width, height)).ok_or_else(|| "no native host GPU".into())
}

/// `VICTORY_TRACE_TARGETS=1`: per frame, what drew into or cleared which title texture, and what
/// was presented (logged at each present).
fn tracing_targets() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TRACE_TARGETS").is_some())
}

/// Draws and clears by title texture since the last present, and dropped texture operations.
static TRACE: Mutex<Option<std::collections::HashMap<usize, [u32; 2]>>> = Mutex::new(None);
static DROPPED_OPS: Mutex<Vec<String>> = Mutex::new(Vec::new());

fn trace_bump(key: usize, which: usize) {
    if tracing_targets() {
        TRACE.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_default().entry(key).or_default()[which] += 1;
    }
}

/// `VICTORY_DUMP_TITLE=<dir>,<seconds>,<seconds>...`: at the first present after each time, the
/// title memory behind every 8-bit colour target drawn to since the last present is read whole,
/// the fraction of non-zero bytes logged, and the memory detiled into a scratch texture and
/// written as `<dir>/m<time>_<n>.ppm`: what the title (CPU, DMA) put there, whatever the GPU did.
#[inline(never)]
fn dump_title_memory(stats: &std::collections::HashMap<usize, [u32; 2]>) {
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    static DONE: Mutex<Vec<u64>> = Mutex::new(Vec::new());
    let start = START.get_or_init(std::time::Instant::now);
    let Ok(v) = std::env::var("VICTORY_DUMP_TITLE") else { return };
    let mut parts = v.split(',');
    let dir = parts.next().unwrap_or(".").to_owned();
    let now = start.elapsed().as_secs();
    let Some(time) = parts
        .filter_map(|p| p.parse::<u64>().ok())
        .find(|&t| now >= t && !DONE.lock().unwrap_or_else(|e| e.into_inner()).contains(&t))
    else {
        return;
    };
    DONE.lock().unwrap_or_else(|e| e.into_inner()).push(time);
    with_title(|t| {
        for (n, key) in stats.keys().enumerate() {
            let Some(info) = crate::resource::texture_info(*key as This) else { continue };
            if !matches!(info.format, 28 | 87 | 91 | 29) {
                continue;
            }
            let tiled = super::textures::read_memory(info.memory, info.size);
            let nz = tiled.iter().filter(|&&b| b != 0).count();
            tracing::info!(time, n, key = format!("{key:#x}"), at = format!("{:#x}", info.memory as usize), size = info.size, nonzero = nz, fraction = nz as f64 / tiled.len().max(1) as f64, "title memory of a colour target");
            let Some(desc) = info.desc() else { continue };
            let Ok(tex) = t.host.texture(&desc) else { continue };
            if super::textures::upload(&t.host.backend, &tex, &info, &tiled).is_err() {
                continue;
            }
            let (w, h) = (info.width as usize, info.height as usize);
            let mut px = vec![0u8; w * h * 4];
            victory_backend::resources::ResourceBackend::read_texture(&t.host.backend, &tex, 0, 0, &mut px, w * 4, w * h * 4);
            let bgra = matches!(info.format, 87 | 91);
            let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
            out.extend(px.as_chunks::<4>().0.iter().flat_map(|p| if bgra { [p[2], p[1], p[0]] } else { [p[0], p[1], p[2]] }));
            let _ = std::fs::write(std::path::Path::new(&dir).join(format!("m{time}_{n}.ppm")), out);
        }
    });
}

/// `VICTORY_DUMP_TARGETS=<dir>,<seconds>`: once, at the first present after that many seconds,
/// every 8-bit colour target drawn to since the last present is written as `<dir>/t<n>.ppm`.
fn dump_targets(stats: &std::collections::HashMap<usize, [u32; 2]>) {
    use std::sync::atomic::AtomicBool;
    static DONE: AtomicBool = AtomicBool::new(false);
    static LAST: Mutex<Option<u64>> = Mutex::new(None);
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    let start = START.get_or_init(std::time::Instant::now);
    let Ok(v) = std::env::var("VICTORY_DUMP_TARGETS") else { return };
    // `<dir>,<seconds>[,<every>]`: from that second on, once or every that many seconds.
    let mut parts = v.split(',');
    let dir = parts.next().unwrap_or(".").to_owned();
    let secs = parts.next().and_then(|s| s.parse::<u64>().ok()).unwrap_or(25);
    let every = parts.next().and_then(|s| s.parse::<u64>().ok());
    let now = start.elapsed().as_secs();
    let due = match (every, *LAST.lock().unwrap_or_else(|e| e.into_inner())) {
        (Some(e), Some(last)) => now >= last + e,
        _ => true,
    };
    if now < secs || !due || (every.is_none() && DONE.swap(true, Ordering::Relaxed)) {
        return;
    }
    *LAST.lock().unwrap_or_else(|e| e.into_inner()) = Some(now);
    with_title(|t| {
        for (n, key) in stats.keys().enumerate() {
            let Some(info) = crate::resource::texture_info(*key as This) else { continue };
            if !matches!(info.format, 28 | 87 | 91 | 29) {
                continue;
            }
            let Some(tex) = t.texture_of(*key) else { continue };
            let (w, h) = (info.width as usize, info.height as usize);
            let mut px = vec![0u8; w * h * 4];
            victory_backend::resources::ResourceBackend::read_texture(&t.host.backend, &tex, 0, 0, &mut px, w * 4, w * h * 4);
            let bgra = matches!(info.format, 87 | 91);
            let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
            out.extend(px.as_chunks::<4>().0.iter().flat_map(|p| if bgra { [p[2], p[1], p[0]] } else { [p[0], p[1], p[2]] }));
            let _ = std::fs::write(std::path::Path::new(&dir).join(format!("t{n}_{now}_{:x}.ppm", info.memory as usize)), out);
            // The alpha channel as grey.
            let mut alpha = format!("P6\n{w} {h}\n255\n").into_bytes();
            alpha.extend(px.as_chunks::<4>().0.iter().flat_map(|p| [p[3], p[3], p[3]]));
            let _ = std::fs::write(std::path::Path::new(&dir).join(format!("t{n}_{now}_{:x}_alpha.ppm", info.memory as usize)), alpha);
            tracing::info!(n, key = format!("{key:#x}"), w, h, "target dumped");
        }
    });
}

/// A texture operation the native host does not do (copies, resolves), counted by the trace.
pub(crate) fn dropped_op(what: &str, dst: This, src: This) {
    crate::list::DROPPED_KINDS[3].fetch_add(1, Ordering::Relaxed);
    warn_once("a texture copy or resolve");
    if tracing_targets() {
        DROPPED_OPS.lock().unwrap_or_else(|e| e.into_inner()).push(format!("{what} {src:?}->{dst:?}"));
    }
}

#[inline(never)]
fn trace_present(resource: This) {
        let stats = TRACE.lock().unwrap_or_else(|e| e.into_inner()).take().unwrap_or_default();
        let ops = std::mem::take(&mut *DROPPED_OPS.lock().unwrap_or_else(|e| e.into_inner()));
        let described: Vec<String> = with_title(|t| {
            stats
                .iter()
                .map(|(k, [d, c])| {
                    let i = crate::resource::texture_info(*k as This);
                    let (f, w, h, a) = i.map_or((0, 0, 0, 0), |i| (i.format, i.width, i.height, i.memory as usize));
                    let mark = if *k == resource as usize { " PRESENTED" } else { "" };
                    let _ = &t;
                    format!("{k:#x} fmt {f} {w}x{h} at {a:#x} draws {d} clears {c}{mark}")
                })
                .collect()
        })
        .unwrap_or_default();
        dump_targets(&stats);
        dump_title_memory(&stats);
        let pi = crate::resource::texture_info(resource);
        tracing::info!(
            present = ?resource,
            at = pi.as_ref().map_or(0, |i| i.memory as usize),
            fmt = pi.as_ref().map_or(0, |i| i.format),
            targets = ?described,
            dropped_ops = ?ops.iter().take(8).collect::<Vec<_>>(),
            dropped = ops.len(),
            "frame targets"
        );
}

/// `PresentX` on title queue `queue`: title texture `resource` over the layer.
pub(crate) fn present(queue: This, resource: This) {
    // Diagnostic: VICTORY_PRESENT_DELAY_MS slows every present, to tell a trigger that follows
    // frames from one that follows time.
    static DELAY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let ms = *DELAY.get_or_init(|| {
        std::env::var("VICTORY_PRESENT_DELAY_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
    });
    if ms > 0 {
        std::thread::sleep(std::time::Duration::from_millis(ms));
    }
    if tracing_targets() {
        trace_present(resource);
    }
    let shown = with_title(|t| {
        let _ = t.ensure_queue(queue as usize);
        t.present(queue as usize, resource as usize)
    });
    match shown {
        Some(Err(e)) => tracing::warn!(error = %e, "a frame was not presented"),
        Some(Ok(false)) => {
            NO_TEXTURE.fetch_add(1, Ordering::Relaxed);
        }
        _ => {}
    }
}

/// Frames presented to the layer so far.
pub fn frames_presented() -> u64 {
    with_title(|t| t.surface().map_or(0, |s| s.frames())).unwrap_or(0)
}

/// The pixels (BGRA) of the last frame presented, once the GPU has drawn it; for tests.
pub fn last_frame() -> Option<(usize, usize, Vec<u8>)> {
    with_title(|t| t.surface().and_then(|s| s.read_last())).flatten()
}

// --- resources and descriptors ---

/// A buffer resource object `res` of `len` bytes at title address `addr`.
pub(crate) fn buffer_created(res: This, addr: u64, len: u64) {
    with_title(|t| t.host.register_buffer(res as usize, addr, len));
}

/// A pipeline object's last reference went: the host pipelines made for it are forgotten, as the
/// title makes thousands of variants (device slot 0x1b8) a minute and a new one often gets a freed
/// one's address (which drew text with the old variant's blend state: glyphs as boxes).
pub(crate) fn pipeline_released(pipeline: This) {
    let key = pipeline as usize;
    with_title(|t| {
        t.pipelines.remove(&key);
        t.console_pipes.retain(|(k, _), _| *k != key);
    });
}

pub(crate) fn resource_released(res: This) {
    with_title(|t| {
        t.host.forget_resource(res as usize);
        t.texture_released(res as usize);
    });
}

/// A title texture object: its host texture is made, and filled from title memory when work is
/// next submitted.
pub(crate) fn texture_created(res: This) {
    if let Some(info) = crate::resource::texture_info(res) {
        with_title(|t| t.texture_created(res as usize, info));
    }
}

/// The GPU writes title texture `key` (a copy destination): realize it now and remember that its
/// contents are the GPU's, not title memory's, so it is never given up.
pub(crate) fn texture_written(key: usize) {
    with_title(|t| {
        t.realize(key);
        t.mark_gpu_written(key);
    });
}

/// `OMSetRenderTargets`: the host textures of the views' resources.
pub(crate) fn targets(this: This, color: &[u64], depth: u64) {
    with_list(this, |t, l| {
        let colors: Vec<usize> = color
            .iter()
            .filter_map(|&h| crate::descriptor::rtv_record(h))
            .map(|r| r.resource as usize)
            .collect();
        let color: Vec<_> = colors.iter().filter_map(|&k| t.texture_of(k)).collect();
        let depth_key = crate::descriptor::dsv_record(depth).map(|r| r.resource as usize);
        let depth = depth_key.and_then(|k| t.texture_of(k));
        l.target_keys = colors.iter().copied().chain(depth_key).collect();
        l.targets = TargetFormats {
            color: colors.iter().filter_map(|&k| t.format_of(k)).collect(),
            depth: depth_key.and_then(|k| t.format_of(k)),
        };
        // Debugging: no depth attachment at all.
        let drop_depth = std::env::var_os("VICTORY_DEBUG_NO_DEPTH_ATTACH").is_some();
        if drop_depth {
            l.targets.depth = None;
        }
        l.list.set_render_targets(&color, depth.as_ref().filter(|_| !drop_depth));
    });
}

/// `ClearRenderTargetView`. shortcut: the whole target, whatever rectangles the title passes.
pub(crate) fn clear_rtv(this: This, handle: u64, color: [f32; 4]) {
    with_list(this, |t, l| {
        let Some(r) = crate::descriptor::rtv_record(handle) else { return };
        let key = r.resource as usize;
        if let Some(tex) = t.texture_of(key) {
            trace_bump(key, 1);
            t.mark_target(key);
            l.list.clear_render_target(&tex, color);
        }
    });
}

#[inline(never)]
fn trace_clear_dsv(flags: u32, depth: f32) {
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n < 6 || (n % 40 == 0 && n < 3000) {
        tracing::info!(flags, depth, n, "ClearDepthStencilView");
    }
}

/// `ClearDepthStencilView`: the depth (flag 1); the stencil is cleared with it.
pub(crate) fn clear_dsv(this: This, handle: u64, flags: u32, depth: f32) {
    if tracing_targets() {
        trace_clear_dsv(flags, depth);
    }
    with_list(this, |t, l| {
        if flags & 1 != 0
            && let Some(r) = crate::descriptor::dsv_record(handle)
        {
            let key = r.resource as usize;
            if let Some(tex) = t.texture_of(key) {
                t.mark_target(key);
                l.list.clear_depth(&tex, depth);
            }
        }
    });
}

/// Shader-visible descriptor heaps by address range: start, end, sampler heap, count. The host
/// heap is made when a list first names the heap, from what the title wrote by then.
static HEAPS: Mutex<Vec<(u64, u64, bool, usize)>> = Mutex::new(Vec::new());

pub(crate) fn heap_created(start: u64, samplers: bool, count: usize) {
    let step = if samplers { 16 } else { 32 };
    HEAPS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((start, start + step * count as u64, samplers, count));
}

pub(crate) fn heap_released(start: u64) {
    HEAPS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .retain(|h| h.0 != start);
    with_title(|t| t.host.drop_heap(start));
}

/// The host heap the descriptor at `handle` is in (made now if it was not) and its index.
fn heap_of(t: &mut MetalTitle, handle: u64) -> Option<(Rc<super::list::NativeHeap<Metal>>, usize)> {
    if let Some(found) = t.host.heap_at(handle) {
        return Some(found);
    }
    let (start, _, samplers, count) = HEAPS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .copied()
        .find(|h| (h.0..h.1).contains(&handle))?;
    let kind = if samplers {
        HeapKind::Sampler
    } else {
        HeapKind::Resource
    };
    let step = if samplers { 16 } else { 32 };
    match t.host.heap(kind, start, count) {
        Ok(heap) => {
            // What the title wrote before the heap was made.
            for i in 0..count as u64 {
                let _ = t.host.mirror(start + i * step);
            }
            t.host.heap_at(handle).or(Some((heap, 0)))
        }
        Err(e) => {
            tracing::warn!(error = %e, "no host descriptor heap");
            None
        }
    }
}

/// The title wrote a view at `handle` (the call `descriptor::mirror` is on Windows).
pub(crate) fn mirror(handle: u64) {
    with_title(|t| {
        match crate::descriptor::view(handle) {
            // A texture the title placed in its own memory has no resource object: the host
            // makes one the first time a view names it.
            Some(crate::descriptor::View::Placed(_, res, d)) if res[0] > 1 => {
                let address = u64::from(d[10]) | u64::from(d[11]) << 32;
                let key = super::textures::placed_key(address, &res);
                if !t.host.knows(key)
                    && let Some(info) = crate::resource::placed_info(address, &res)
                {
                    t.texture_created(key, info);
                    t.realize(key);
                }
                t.mark_used(key);
            }
            // A view of a texture object that no list has used yet.
            Some(
                crate::descriptor::View::Shader(r, _) | crate::descriptor::View::Unordered(r, ..),
            ) => {
                t.realize(r);
                t.mark_used(r);
            }
            _ => {}
        }
        if let Err(e) = t.host.mirror(handle) {
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 3000 {
                tracing::info!(error = %e, handle = format!("{handle:#x}"), "a descriptor was not mirrored");
            }
        }
    });
}

// --- list recording ---

pub(crate) fn set_pipeline(this: This, pipeline: This) {
    with_list(this, |_, l| l.pipeline = pipeline as usize);
}

/// The host pipeline of title pipeline `key` for a draw into `targets`, made at first use. A
/// console pipeline's state names no target formats, so it is made again for each set of targets.
fn pipe(t: &mut MetalTitle, key: usize, targets: &TargetFormats) -> Pipe<Metal> {
    if let Some(p) = t.pipelines.get(&key) {
        return p.clone();
    }
    let console_key = (key, targets.clone());
    if let Some(p) = t.console_pipes.get(&console_key) {
        return p.clone();
    }
    let (made, console) = make_pipe(t, key, targets);
    if console {
        t.console_pipes.insert(console_key, made.clone());
    } else {
        t.pipelines.insert(key, made.clone());
    }
    made
}

/// Work dropped because its pipeline has no host pipeline.
static DROPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Counts a draw or dispatch the host drops, and warns the first time for each `why`.
fn dropped(why: &'static str) {
    DROPPED.fetch_add(1, Ordering::Relaxed);
    warn_once(why);
}

/// A warning logged once per distinct `what` for the run.
pub(crate) fn warn_once(what: &'static str) {
    static SEEN: std::sync::Mutex<Vec<&'static str>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&what) {
        seen.push(what);
        tracing::warn!("native host drops work: {what} (first time; counted in \"native draws so far\")");
    }
}

/// Draws the title recorded, and those that went on to the host list.
static RECORDED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ENCODED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Presents whose resource had no host texture.
static NO_TEXTURE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `(draws recorded, draws handed to the host list, draws dropped for want of a host pipeline,
/// presents that found no host texture)`, for the frame log.
pub(crate) fn draw_counts() -> (u64, u64, u64, u64) {
    (
        RECORDED.load(Ordering::Relaxed),
        ENCODED.load(Ordering::Relaxed),
        DROPPED.load(Ordering::Relaxed),
        NO_TEXTURE.load(Ordering::Relaxed),
    )
}

/// The frame log's line of native draw counters, and the commands dropped by kind (kept out of
/// `PresentX`, whose frame it would otherwise enlarge).
#[inline(never)]
pub(crate) fn log_draw_counts(frame: u64) {
    let (recorded, encoded, no_pipeline, no_texture) = draw_counts();
    let (mesh, mesh_indirect, _, copies, _) = crate::list::dropped_kinds();
    let indirect = crate::list::indirect_dropped();
    tracing::info!(frame, recorded, encoded, no_pipeline, no_texture, mesh, mesh_indirect, indirect, texture_copies = copies, "native draws so far");
}

/// A title pipeline object's host pipeline: its shaders translated for the host, then converted.
/// The second result says the pipeline depends on the draw's targets.
fn make_pipe(t: &mut MetalTitle, key: usize, targets: &TargetFormats) -> (Pipe<Metal>, bool) {
    let Some((stages, graphics)) = crate::pipeline::native_stages(key as This) else {
        return (Pipe::Refused, false);
    };
    let graphics = graphics.map(|g| {
        if g.names_targets() {
            (g, false)
        } else {
            (super::console_state::graphics_state_for(&g, targets.color.clone(), targets.depth), true)
        }
    });
    let console = graphics.as_ref().is_some_and(|g| g.1);
    if console && std::env::var_os("VICTORY_TRACE_BIND").is_some() {
        tracing::info!(pipeline = key, ?targets, state = ?graphics.as_ref().map(|g| &g.0), "console pipeline state at its draw");
    }
    let graphics = graphics.map(|g| g.0);
    (make_host_pipe(t, stages, graphics), console)
}

fn make_host_pipe(
    t: &mut MetalTitle,
    stages: Vec<(&'static str, Arc<crate::pipeline::Shader>)>,
    graphics: Option<super::state::Graphics>,
) -> Pipe<Metal> {
    if stages.is_empty() && graphics.as_ref().is_some_and(|g| g.is_guessed()) {
        return Pipe::Skipped;
    }
    let name = stages
        .iter()
        .map(|(stage, s)| format!("{stage} {}", s.hash))
        .collect::<Vec<_>>()
        .join(" ");
    let mut hosts = Vec::new();
    for (stage, shader) in &stages {
        match shader.native() {
            Ok(h) => hosts.push((*stage, h.clone())),
            Err(e) => {
                t.refused_translation(&name, e.to_string());
                return Pipe::Refused;
            }
        }
    }
    let find = |stage: &str| hosts.iter().find(|h| h.0 == stage).map(|h| &*h.1);
    match graphics {
        None => match find("compute") {
            Some(cs) => t.compute_pipe(cs, &name),
            None => Pipe::Refused,
        },
        Some(g) if g.is_guessed() => match hosts.iter().find(|h| h.0 == "vertex") {
            Some((_, vs)) => Pipe::Guess(Rc::new(Guessed {
                vs_source: stages.iter().find(|s| s.0 == "vertex").expect("a vertex stage").1.clone(),
                vs: vs.clone(),
                ps: hosts.iter().find(|h| h.0 == "pixel").map(|h| h.1.clone()),
                name,
                state: g.clone(),
                made: Default::default(),
            })),
            None => Pipe::Refused,
        },
        Some(g) => match find("vertex") {
            Some(vs) => t.render_pipe(vs, find("pixel"), &g, &name),
            None => {
                t.refused_translation(&name, "a graphics pipeline without a vertex shader".into());
                Pipe::Refused
            }
        },
    }
}

/// The shader path's counters: pipelines made and cached, refusals by reason.
pub fn shader_stats() -> Option<super::shaders::ShaderStats> {
    with_title(|t| t.shaders.clone())
}

/// Makes title pipeline `key` the compute pipeline of a pre-translated DXIL container with the
/// given group size (what a translated shader's pipeline is made of); for diagnostics and tests.
pub fn install_compute_pipeline(key: usize, dxil: &[u8], threads: [usize; 3]) -> Result<(), String> {
    with_title(|t| {
        let pso = t.host.compute_pipeline(dxil, threads)?;
        t.pipelines.insert(key, Pipe::Compute(pso));
        Ok(())
    })
    .unwrap_or_else(|| Err("no native host GPU".into()))
}

pub(crate) fn dispatch(this: This, x: u32, y: u32, z: u32) {
    with_list(this, |t, l| match pipe(t, l.pipeline, &TargetFormats::default()) {
        Pipe::Compute(pso) => {
            l.list.set_compute_pipeline(&pso);
            if let Err(e) = l.list.dispatch(x, y, z) {
                tracing::warn!(error = %e, "dispatch");
            }
        }
        _ => dropped("a dispatch without a host compute pipeline"),
    });
}

/// `IASetPrimitiveTopology`.
pub(crate) fn topology(this: This, topology: u32) {
    with_list(this, |_, l| l.topology = topology);
}

/// `RSSetViewports`: the first one.
pub(crate) fn viewports(this: This, first: Option<[f32; 6]>) {
    if tracing_targets() {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 8 {
            tracing::info!(?first, "viewport set");
        }
    }
    with_list(this, |_, l| {
        l.list.set_viewport(first.map(|v| victory_backend::resources::Viewport {
            x: v[0],
            y: v[1],
            width: v[2],
            height: v[3],
            min_depth: v[4],
            max_depth: v[5],
        }))
    });
}

/// `RSSetScissorRects`: the first one (left, top, right, bottom).
pub(crate) fn scissors(this: This, first: Option<[i32; 4]>) {
    with_list(this, |_, l| l.list.set_scissor(first));
}

pub(crate) fn blend_factor(this: This, factor: Option<[f32; 4]>) {
    with_list(this, |_, l| l.list.set_blend_factor(factor.unwrap_or([1.0; 4])));
}

pub(crate) fn stencil_ref(this: This, reference: u32) {
    with_list(this, |_, l| l.list.set_stencil_ref(reference));
}

/// `IASetIndexBuffer`: the buffer's address and whether its indices are 32 bits.
pub(crate) fn index_buffer(this: This, view: Option<(u64, bool)>) {
    with_list(this, |_, l| match view {
        Some((address, wide)) => l.list.set_index_buffer(address, wide),
        None => l.list.clear_index_buffer(),
    });
}

/// `IASetVertexBuffers`: the addresses of the streams from `start`.
pub(crate) fn vertex_buffers(this: This, start: u32, addresses: &[u64], strides: &[u32]) {
    with_list(this, |_, l| {
        for (i, a) in addresses.iter().enumerate() {
            l.list.set_vertex_buffer(start as usize + i, *a);
            let slot = start as usize + i;
            if l.strides.len() <= slot {
                l.strides.resize(slot + 1, 0);
            }
            l.strides[slot] = strides.get(i).copied().unwrap_or(0);
        }
    });
}

/// The host pipeline of a guessed pipeline for the targets and vertex buffers the list has bound.
fn guess(t: &mut MetalTitle, l: &MetalList, g: &Guessed<Metal>, rect: bool) -> Pipe<Metal> {
    let key = format!("{:?}{rect}", l.strides);
    if let Some(p) = g.made.borrow().get(&key) {
        return p.clone();
    }
    let rect_vs = if rect {
        match g.vs_source.native_rect() {
            Ok(h) => Some(h.clone()),
            Err(e) => {
                t.refused_translation(&g.name, format!("rectangle list: {e}"));
                return Pipe::Refused;
            }
        }
    } else {
        None
    };
    // Parameters the pixel shader reads and the vertex shader does not write: written (undefined)
    // too, as the converter refuses a pipeline whose fragment inputs the vertex stage lacks.
    let base_vs = rect_vs.as_deref().unwrap_or(&g.vs);
    let extra = g.ps.as_ref().map_or(0, |ps| {
        super::state::texcoords(&ps.dxil, victory_shader::FourCC::ISG1)
            & !super::state::texcoords(&base_vs.dxil, victory_shader::FourCC::OSG1)
    });
    let widened = if extra != 0 {
        let mut copy = [0u8; 32];
        for &(orig, new) in g.ps.as_ref().map_or(&[][..], |ps| &ps.splits[..]) {
            if orig < 32 && new < 32 {
                copy[new as usize] = orig as u8 + 1;
            }
        }
        match g.vs_source.translate_extra(rect, extra, copy) {
            Ok(h) => Some(h),
            Err(e) => {
                t.refused_translation(&g.name, format!("extra outputs: {e}"));
                return Pipe::Refused;
            }
        }
    } else {
        None
    };
    let vs = widened.as_deref().unwrap_or(base_vs);
    let reads = super::state::vertex_inputs(&vs.dxil).unwrap_or_default();
    tracing::info!(name = %g.name, ?reads, targets = ?l.targets, strides = ?l.strides, "guessed pipeline");
    let state = g.state.for_draw(l.targets.color.clone(), l.targets.depth, l.strides.clone(), &reads, rect);
    if tracing_targets() {
        tracing::info!(name = %g.name, ?state, depth_test = ?state.depth_test(), stencil = ?state.stencil(), cull = ?state.cull(), color = ?state.color_state(), "guessed pipeline state");
    }
    let made = t.render_pipe(vs, g.ps.as_deref(), &state, &g.name);
    g.made.borrow_mut().insert(key, made.clone());
    made
}

/// Sets the draw pipeline, topology and bindings. Returns rectangle mode for a drawable pipeline.
fn graphics(t: &mut MetalTitle, l: &mut MetalList) -> Option<bool> {
    // A rectangle list (topology 17) of a guessed pipeline is drawn as instances of four
    // vertices, a triangle strip, each instance a rectangle (`Options::rect_list`).
    let rect = l.topology == 17;
    let targets = l.targets.clone();
    let made = match pipe(t, l.pipeline, &targets) {
        Pipe::Guess(g) => guess(t, l, &g, rect),
        p if rect => {
            if std::env::var_os("VICTORY_TRACE_RECTS").is_some() {
                tracing::info!(pipeline = ?l.pipeline, kind = ?std::mem::discriminant(&p), "rect draw of a pipeline that is not guessed");
            }
            warn_once("a rectangle-list draw of a described pipeline, drawn without its fourth corner");
            p
        }
        Pipe::Skipped => {
            warn_once("a draw of a guessed pipeline without a vertex stage (a mesh pipeline)");
            return None;
        }
        p => p,
    };
    let Pipe::Render(pso, binding) = made else {
        dropped("a draw without a host render pipeline (refused or not translated)");
        return None;
    };
    let topology = if rect {
        Some(victory_backend::resources::Topology::TriangleStrip)
    } else {
        super::state::topology(l.topology)
    };
    let Some(topology) = topology else {
        static SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if l.topology < 64 && SEEN.fetch_or(1 << l.topology, Ordering::Relaxed) >> l.topology & 1 == 0 {
            tracing::warn!(topology = l.topology, "a draw dropped for its primitive topology");
        }
        dropped("a draw of a primitive topology the host does not draw");
        return None;
    };
    l.list.set_graphics_pipeline(&pso, topology);
    l.list.set_graphics_binding(binding);
    Some(rect)
}

pub(crate) fn draw(this: This, vertices: u32, instances: u32, first: u32, first_instance: u32) {
    RECORDED.fetch_add(1, Ordering::Relaxed);
    if tracing_targets() {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        static START: OnceLock<std::time::Instant> = OnceLock::new();
        let late = START.get_or_init(std::time::Instant::now).elapsed().as_secs() >= 26;
        let n = N.fetch_add(1, Ordering::Relaxed);
        if n < 8 || (late && n % 7 == 0 && n < 4000) {
            with_list(this, |_, l| {
                let mut head = [0u8; 72];
                let at = l.list.vertex_buffer_address(0);
                let ok = at.is_some_and(|a| super::safe_read(a as *const u8, &mut head));
                tracing::info!(vertices, instances, first, topology = l.topology, vb0 = ?at, ok, head = ?head, strides = ?l.strides, formats = ?l.targets, "draw");
            });
        }
    }
    // Debugging: only indexed draws (the content), not the boot passes.
    if std::env::var_os("VICTORY_DEBUG_ONLY_INDEXED").is_some() {
        return;
    }
    with_list(this, |t, l| {
        if let Some(rect) = graphics(t, l) {
            if rect && std::env::var_os("VICTORY_TRACE_RECTS").is_some() {
                static SEEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
                let mut seen = SEEN.lock().unwrap();
                if seen.iter().filter(|&&p| p == l.pipeline as usize).count() < 3 {
                    seen.push(l.pipeline as usize);
                    let stride = l.strides.first().copied().unwrap_or(0) as usize;
                    let mut head = vec![0u8; (stride * 3).clamp(12, 192)];
                    let at = l.list.vertex_buffer_address(0).map(|a| a + (first as u64) * stride as u64);
                    let ok = at.is_some_and(|a| super::safe_read(a as *const u8, &mut head));
                    let floats: Vec<f32> = head.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
                    tracing::info!(pipeline = ?l.pipeline, vertices, first, strides = ?l.strides, ok, ?floats, "rect draw");
                }
            }
            let call = if rect {
                l.list.draw_instanced(4, vertices / 3, 0, first / 3)
            } else {
                l.list.draw_instanced(vertices, instances, first, first_instance)
            };
            match call {
                Ok(()) => {
                    ENCODED.fetch_add(1, Ordering::Relaxed);
                    for &k in &l.target_keys {
                        t.mark_target(k);
                        trace_bump(k, 0);
                    }
                }
                Err(e) => tracing::warn!(error = %e, "draw"),
            }
        }
    });
}

pub(crate) fn draw_indexed(this: This, indices: u32, instances: u32, first: u32, base: i32, first_instance: u32) {
    RECORDED.fetch_add(1, Ordering::Relaxed);
    with_list(this, |t, l| {
        if let Some(rect) = graphics(t, l) {
            if rect {
                // shortcut: indexed rectangle lists are not drawn.
                dropped("an indexed rectangle-list draw");
                return;
            }
            match l.list.draw_indexed_instanced(indices, instances, first, base, first_instance) {
                Ok(()) => {
                    ENCODED.fetch_add(1, Ordering::Relaxed);
                    for &k in &l.target_keys {
                        t.mark_target(k);
                        trace_bump(k, 0);
                    }
                }
                Err(e) => tracing::warn!(error = %e, "indexed draw"),
            }
        }
    });
}

pub(crate) fn root_constants(this: This, compute: bool, slot: u32, offset: u32, words: &[u32]) {
    with_list(this, |_, l| {
        l.list
            .set_root_constants_at(compute, slot as usize, offset as usize, words)
    });
}

pub(crate) fn root_view(this: This, compute: bool, slot: u32, address: u64) {
    with_list(this, |_, l| {
        if compute {
            l.list.set_compute_root_view(slot as usize, address)
        } else {
            l.list.set_graphics_root_view(slot as usize, address)
        }
    });
}

pub(crate) fn root_table(this: This, compute: bool, slot: u32, handle: u64) {
    with_list(this, |t, l| {
        let Some((heap, index)) = heap_of(t, handle) else {
            return;
        };
        if compute {
            l.list.set_compute_root_table(slot as usize, &heap, index)
        } else {
            l.list.set_graphics_root_table(slot as usize, &heap, index)
        }
    });
}

/// `SetDescriptorHeaps`: the title's heap objects.
pub(crate) fn set_heaps(this: This, heaps: &[This]) {
    with_list(this, |t, l| {
        let (mut views, mut samplers) = (None, None);
        for &h in heaps {
            let Some((start, sampler, _)) = crate::descriptor::span(h) else {
                continue;
            };
            let Some((heap, _)) = heap_of(t, start) else {
                continue;
            };
            if sampler {
                samplers = Some(heap);
            } else {
                views = Some(heap);
            }
        }
        l.list.set_heaps(views.as_ref(), samplers.as_ref());
    });
}

/// `CopyBufferRegion` between title buffers.
pub(crate) fn copy_buffer(this: This, dst: (This, u64), src: (This, u64), len: u64) {
    let (Some((to, to_size)), Some((from, from_size))) = (
        crate::resource::buffer_bytes(dst.0),
        crate::resource::buffer_bytes(src.0),
    ) else {
        return;
    };
    if dst.1.saturating_add(len) > to_size || src.1.saturating_add(len) > from_size {
        tracing::warn!(len, "copy outside its buffer");
        return;
    }
    // SAFETY: inside the buffers, checked above.
    let (to, from) = unsafe { (to.add(dst.1 as usize), from.add(src.1 as usize)) };
    copy_memory(this, to as u64, from as u64, len);
}

/// `CopyMemoryToMemoryX` and buffer copies: title addresses.
pub(crate) fn copy_memory(this: This, dst: u64, src: u64, len: u64) {
    with_list(this, |_, l| {
        l.list.copy_buffer_region(dst, src, len as usize);
    });
}

/// A bottom-of-pipe value write of `value`, `bytes` long.
pub(crate) fn write_value(this: This, address: u64, value: u64, bytes: usize) -> bool {
    if std::env::var_os("VICTORY_TRACE_WRITES").is_some() {
        let inside = pools().resolve(address, bytes as u64).is_some();
        tracing::warn!("list write {value:#x} ({bytes} bytes) at {address:#x}, in a pool: {inside}");
    }
    with_list(this, |_, l| l.list.write_value(address, value, bytes)).is_some()
}

/// A wait for a memory value, recorded at the current position of the list.
pub(crate) fn wait_value(this: This, w: Wait) {
    with_list(this, |_, l| l.wait(w));
}

/// `Reset`: forgets what was recorded (the host objects it named go with it, under the lock).
pub(crate) fn reset_list(this: This) {
    with_list(this, |_, l| l.reset());
}

/// Drops a list object, whose `Rc`s must be dropped under the lock.
pub(crate) fn locked<R>(f: impl FnOnce() -> R) -> R {
    let mut f = Some(f);
    match with_title(|_| (f.take().expect("once"))()) {
        Some(r) => r,
        None => (f.take().expect("not run"))(),
    }
}

/// Counters of the host for tests and the frame line.
pub fn stats() -> Option<super::title::Stats> {
    with_title(|t| t.stats.clone())
}
