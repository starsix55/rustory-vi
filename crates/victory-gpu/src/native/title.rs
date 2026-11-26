//! What the title-facing objects (`queue.rs`, `fence.rs`, `list.rs`) need from the native host,
//! generic over the backend: queues and fences by the title's object address, title lists cut at
//! memory-value waits, and `hold.rs`'s per-queue FIFO of held work in front of
//! [`NativeHost::execute`]. The macOS glue (`glue.rs`) instantiates it once with Metal.
//!
//! Ordering. Work on one title queue is one backend queue, so submissions run in the order they
//! were made. A list segment that waits for a memory value another queue's list writes stays on
//! the CPU side until the value is there (`hold.rs`; the GPU's own writes into title memory are
//! visible to the CPU as they land, there is no copy); everything else goes to the GPU at once and
//! is never waited for. Fence signals and waits that need no CPU decision are encoded on the GPU
//! (`MTLSharedEvent`); behind held work they are held too, so that they stay in order.

use super::host::NativeHost;
use super::list::{Binding, ComputePso, NativeHeap, NativeList};
use super::memory::PoolTable;
use crate::hold::{Held, Item, ValueWait, World};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use victory_backend::backend::ComputeBackend;
use victory_backend::resources::{Format, GraphicsBackend, PresentBackend};

/// A wait for a memory value (`Wait32BitValueX`, `Wait64BitValueX`): the value at `address`
/// (`bytes` 4 or 8, masked) compared with `reference` by a D3D12 comparison function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wait {
    pub address: u64,
    pub bytes: u8,
    pub comparison: u32,
    pub reference: u64,
    pub mask: u64,
}

impl From<Wait> for ValueWait {
    fn from(w: Wait) -> ValueWait {
        ValueWait {
            address: w.address,
            bytes: w.bytes,
            comparison: w.comparison,
            reference: w.reference,
            mask: w.mask,
        }
    }
}

/// A title list: the recording [`NativeList`] plus the segments cut off it at waits recorded after
/// commands, each with the waits at its start.
pub struct TitleList<B: GraphicsBackend> {
    done: Vec<(NativeList<B>, Vec<Wait>)>,
    /// The list the calls record into, and the waits at its start.
    pub list: NativeList<B>,
    waits: Vec<Wait>,
    /// Waits recorded: the ones that cut a segment.
    pub cuts: u32,
    /// The title pipeline object `SetPipelineState` named last (0: none), and the primitive
    /// topology set (the console's `D3D_PRIMITIVE_TOPOLOGY`).
    pub pipeline: usize,
    pub topology: u32,
    /// The formats of the targets `OMSetRenderTargets` bound last.
    pub targets: TargetFormats,
    /// The strides of the bound vertex buffers by slot, which a guessed pipeline is made from at
    /// its draw, and the title resources bound as targets (for the target trace).
    pub strides: Vec<u32>,
    pub target_keys: Vec<usize>,
}

/// The formats of a draw's bound colour and depth targets.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct TargetFormats {
    pub color: Vec<Format>,
    pub depth: Option<Format>,
}

impl<B: GraphicsBackend> Default for TitleList<B> {
    fn default() -> Self {
        TitleList {
            done: Vec::new(),
            list: NativeList::default(),
            waits: Vec::new(),
            cuts: 0,
            pipeline: 0,
            topology: 0,
            targets: TargetFormats::default(),
            strides: Vec::new(),
            target_keys: Vec::new(),
        }
    }
}

impl<B: GraphicsBackend> TitleList<B> {
    /// `Reset`.
    pub fn reset(&mut self) {
        *self = TitleList::default();
    }

    /// A wait recorded at the current position: at the start of the segment if it has no commands
    /// yet, else the segment ends here and the wait starts the next one (state carries over).
    pub fn wait(&mut self, w: Wait) {
        if !self.list.is_empty() {
            let segment = self.list.cut();
            self.done
                .push((segment, std::mem::replace(&mut self.waits, vec![w])));
            self.cuts += 1;
        } else {
            self.waits.push(w);
        }
    }

    /// The segments a submission runs, in order (copies: the title may reset and re-record).
    pub fn segments(&self) -> Vec<(NativeList<B>, Vec<Wait>)> {
        self.done
            .iter()
            .map(|(l, w)| (l.snapshot(), w.clone()))
            .chain([(self.list.snapshot(), self.waits.clone())])
            .collect()
    }
}

/// Counters for the frame line and the tests.
#[derive(Default, Debug, Clone)]
pub struct Stats {
    /// Segments queued by `ExecuteCommandLists`.
    pub segments: u64,
    /// Groups of segments handed to the GPU (a piece is what `hold` runs together).
    pub pieces: u64,
    /// Pieces the host refused (an address outside the mapped pools, ...): logged, their fences
    /// are signalled all the same.
    pub refused: u64,
    /// Segments that waited for a memory value before they ran.
    pub waited: u64,
    /// Title textures detiled (again) because their memory changed.
    pub detiles: u64,
    /// Heaps realized conservatively (whole heap) at encode time.
    pub realize_whole: u64,
    /// Heaps realized from analyzed descriptor ranges at encode time.
    pub realize_ranged: u64,
    /// Pending textures realized because a fixed descriptor range named them.
    pub realize_required: u64,
    /// Pending textures realized only because a computed index forced the heap.
    pub realize_conservative: u64,
    /// Pending textures left for a later command because no analyzed range named them.
    pub realize_deferred: u64,
    /// Textures whose layout the host cannot detile (counted once per content).
    pub undetilable: u64,
}

/// What a title pipeline object became on the host.
pub enum Pipe<B: GraphicsBackend> {
    Compute(Rc<ComputePso<B>>),
    /// With the argument layout of its translated shaders (none: the title's root as it is).
    Render(Rc<B::RenderPipeline>, Option<Rc<Binding>>),
    /// A guessed graphics pipeline (`VICTORY_GUESS_GRAPHICS`): made, per target formats and
    /// vertex strides, at its draws.
    Guess(Rc<Guessed<B>>),
    /// A guessed graphics slot without a vertex producer.
    #[cfg(target_os = "macos")]
    Skipped,
    /// The host refused it (the reason is logged and counted once); work using it is dropped.
    Refused,
}

impl<B: GraphicsBackend> Clone for Pipe<B> {
    fn clone(&self) -> Self {
        match self {
            Pipe::Compute(p) => Pipe::Compute(p.clone()),
            Pipe::Render(p, b) => Pipe::Render(p.clone(), b.clone()),
            Pipe::Guess(g) => Pipe::Guess(g.clone()),
            #[cfg(target_os = "macos")]
            Pipe::Skipped => Pipe::Skipped,
            Pipe::Refused => Pipe::Refused,
        }
    }
}

/// The shaders of a guessed pipeline and the host pipelines made from them so far.
pub struct Guessed<B: GraphicsBackend> {
    pub vs: Arc<victory_shader::emit::HostShader>,
    pub ps: Option<Arc<victory_shader::emit::HostShader>>,
    pub name: String,
    /// The pipeline's state with the targets of its draw ([`Graphics::for_draw`] completes it).
    pub state: super::state::Graphics,
    /// The vertex shader's container, translated again for rectangle lists at the first such draw.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub(crate) vs_source: Arc<crate::pipeline::Shader>,
    /// By the formats and strides they were made for.
    pub made: std::cell::RefCell<HashMap<String, Pipe<B>>>,
}

pub struct Title<B: PresentBackend> {
    pub host: NativeHost<B>,
    /// Host pipelines by title pipeline object, made at first use.
    pub pipelines: HashMap<usize, Pipe<B>>,
    /// Host pipelines of the console's pipelines (whose state names no target formats) by title
    /// pipeline object and the targets of the draw ([`super::console_state`]).
    pub console_pipes: HashMap<(usize, TargetFormats), Pipe<B>>,
    /// What the shader path did, and refused.
    pub shaders: super::shaders::ShaderStats,
    pub(super) libraries: super::shaders::Libraries<B>,
    pub(super) textures: super::textures::Textures<B>,
    /// Sampled textures realized but not yet uploaded once.
    pub(super) unloaded: std::collections::HashSet<usize>,
    /// Keys of textures currently in the fast per-submission probe set (`hot > 0`).
    pub(super) hot_keys: Vec<usize>,
    /// Round-robin cursor into `hot_keys` when more than `MAX_SYNC_HOT` textures are hot.
    pub(super) hot_cursor: usize,
    /// Reusable scratch buffer for hot-texture probes (avoids per-submission allocations).
    pub(super) probe_buf: Vec<u8>,
    /// Where `Present` shows frames, if the app gave a layer.
    pub(super) surface: Option<B::Surface>,
    /// Title textures no view has named yet.
    pub(super) pending: HashMap<usize, super::textures::TextureInfo>,
    /// Host textures by canonical title-memory range and layout. Identical
    /// aliases share one Metal texture; `Weak` lets the last user free it.
    pub(super) backing: HashMap<
        (usize, usize, u32, u32, u32, u32),
        (
            super::textures::TextureInfo,
            std::rc::Weak<B::Texture>,
        ),
    >,
    /// The last frame texture eviction ran: it is O(realized textures), so
    /// once a frame rather than once a submission.
    pub(super) last_evict: u64,
    queues: HashMap<usize, Rc<B::Queue>>,
    fences: HashMap<usize, Rc<B::Fence>>,
    held: Held<Rc<NativeList<B>>>,
    /// Signals held behind work: the queue, fence and value, by the number the held item carries.
    signals: HashMap<usize, (usize, usize, u64)>,
    next_signal: usize,
    /// The highest value each fence was told to reach, and when that changed (diagnostics).
    requested: HashMap<usize, (u64, std::time::Instant)>,
    pub stats: Stats,
}

impl<B: PresentBackend> Title<B> {
    pub fn new(backend: B, memory: Arc<PoolTable>) -> Self {
        Title {
            host: NativeHost::with_memory(backend, memory),
            pipelines: HashMap::new(),
            console_pipes: HashMap::new(),
            shaders: Default::default(),
            libraries: Default::default(),
            textures: Default::default(),
            backing: Default::default(),
            last_evict: u64::MAX,
            unloaded: std::collections::HashSet::new(),
            hot_keys: Vec::new(),
            hot_cursor: 0,
            probe_buf: Vec::new(),
            surface: None,
            pending: HashMap::new(),
            queues: HashMap::new(),
            fences: HashMap::new(),
            requested: HashMap::new(),
            held: Held::new(),
            signals: HashMap::new(),
            next_signal: 0,
            stats: Stats::default(),
        }
    }

    fn queue(&mut self, key: usize) -> Result<Rc<B::Queue>, String> {
        if let Some(q) = self.queues.get(&key) {
            return Ok(q.clone());
        }
        let q = Rc::new(self.host.backend.create_queue()?);
        self.queues.insert(key, q.clone());
        Ok(q)
    }

    /// The backend fence of title fence `key`, made with `initial` the first time.
    pub fn fence(&mut self, key: usize, initial: u64) -> Result<Rc<B::Fence>, String> {
        if let Some(f) = self.fences.get(&key) {
            return Ok(f.clone());
        }
        let f = Rc::new(self.host.backend.create_fence(initial)?);
        self.fences.insert(key, f.clone());
        Ok(f)
    }

    /// The title released fence `key`.
    pub fn forget_fence(&mut self, key: usize) {
        self.fences.remove(&key);
    }

    /// The title released queue `key`; held work stays held (the title does not do it).
    pub fn forget_queue(&mut self, key: usize) {
        self.queues.remove(&key);
    }

    /// Every title fence with a backend fence and the value it has completed: what the title's own
    /// copy of the value (read from memory) is raised to.
    pub fn completed(&self) -> Vec<(usize, u64)> {
        self.fences
            .iter()
            .map(|(k, f)| (*k, self.host.backend.completed(f)))
            .collect()
    }

    /// Fence `key`'s completed value (made with `initial` if it has no backend fence yet).
    pub fn fence_value(&mut self, key: usize, initial: u64) -> u64 {
        match self.fence(key, initial) {
            Ok(f) => self.host.backend.completed(&f).max(initial),
            Err(_) => initial,
        }
    }

    /// `ID3D12Fence::Signal`: the value is reached now; queues waiting for it go on.
    pub fn signal_cpu(&mut self, key: usize, initial: u64, value: u64) {
        if let Ok(f) = self.fence(key, initial) {
            self.host.backend.signal(&f, value);
        }
        self.pump();
    }

    /// Whether fence `key` reached `value`; pumps held work first, so that a wait for a fence
    /// signalled behind a memory-value wait can make progress when the caller polls.
    pub fn try_wait(&mut self, key: usize, initial: u64, value: u64) -> bool {
        self.pump();
        self.fence_value(key, initial) >= value
    }

    /// `ExecuteCommandLists` on title queue `queue`: the lists' segments join what the queue holds
    /// and whatever may run goes to the GPU.
    /// Whether work that last named `heap` may still be running. Descriptor
    /// entries may only be repointed after it completes.
    pub(super) fn heap_in_flight(&self, heap: &NativeHeap<B>) -> bool {
        let queue = heap.last_queue.get();
        let Some(q) = self.queues.get(&queue) else {
            return false;
        };
        let (_, completed) = self.host.backend.queue_progress(&**q);
        heap.last_serial.get() > completed
    }

    pub fn submit(&mut self, queue: usize, segments: Vec<(NativeList<B>, Vec<Wait>)>) {        self.sync_dirty();
        for (list, waits) in segments {
            self.stats.segments += 1;
            let waits: Vec<ValueWait> = waits.into_iter().map(Into::into).collect();
            self.held.push(queue, Item::Segment(Rc::new(list), waits));
        }
        self.pump();
    }

    /// `ID3D12CommandQueue::Signal`: behind work the queue still holds, else on the GPU at once
    /// (after the work already submitted to its queue).
    pub fn signal_queue(&mut self, queue: usize, fence: usize, initial: u64, value: u64) {
        let r = self.requested.entry(fence).or_insert((0, std::time::Instant::now()));
        if value > r.0 {
            *r = (value, std::time::Instant::now());
        }
        if self.held.holds_work(queue) {
            let n = self.next_signal;
            self.next_signal += 1;
            self.signals.insert(n, (queue, fence, value));
            // Make the backend fence now: `initial` is its value until the signal runs.
            let _ = self.fence(fence, initial);
            self.held.push(queue, Item::Signal(n, 0));
        } else {
            self.signal_gpu(queue, fence, initial, value);
        }
        self.pump();
    }

    fn signal_gpu(&mut self, queue: usize, fence: usize, initial: u64, value: u64) {
        let (Ok(q), Ok(f)) = (self.queue(queue), self.fence(fence, initial)) else {
            return;
        };
        if let Err(e) = self.host.execute(&q, &[], &[], &[(&*f, value)]) {
            tracing::warn!(error = %e, "a fence signal was not queued");
        }
    }

    /// `ID3D12CommandQueue::Wait`: later work on the queue starts once the fence has the value.
    pub fn wait_queue(&mut self, queue: usize, fence: usize, initial: u64, value: u64) {
        if self.held.holds_work(queue) {
            let _ = self.fence(fence, initial);
            self.held.push(queue, Item::FenceWait(fence, value));
        } else if let (Ok(q), Ok(f)) = (self.queue(queue), self.fence(fence, initial))
            && let Err(e) = self.host.execute(&q, &[], &[(&*f, value)], &[])
        {
            tracing::warn!(error = %e, "a fence wait was not queued");
        }
        self.pump();
    }

    /// Runs the heads of held queues that may run, until none can.
    pub fn pump(&mut self) {
        if self.held.is_empty() {
            return;
        }
        let Title {
            host,
            queues,
            fences,
            held,
            signals,
            stats,
            textures,
            pending,
            backing,
            unloaded,
            ..
        } = self;
        let memory = host.memory();
        let read = |address: u64, bytes: u8| read_title(&memory, address, bytes);
        let fence = |f: usize| {
            fences
                .get(&f)
                .map_or(u64::MAX, |fence| host.backend.completed(fence))
        };
        let mut run = |queue: usize, lists: Vec<Rc<NativeList<B>>>| {
            stats.pieces += 1;
            let Some(q) = queues.get(&queue).cloned() else {
                return;
            };
            let (submitted, _) = host.backend.queue_progress(&*q);
            let heaps = match super::textures::realize_pending(
                host,
                pending,
                textures,
                backing,
                unloaded,
                stats,
                queue,
                submitted,
                &lists,
            ) {
                Ok(heaps) => heaps,
                Err(e) => {
                    stats.refused += 1;
                    tracing::error!(
                        error = %e,
                        "refusing a piece: a texture could not be realized; no placeholder rendering"
                    );
                    if std::env::var_os("VICTORY_ALLOW_PLACEHOLDERS").is_none() {
                        tracing::error!("stopping the diagnostic run on the failed realization");
                        std::process::exit(4);
                    }
                    Vec::new()
                }
            };
            let refs: Vec<&NativeList<B>> = lists.iter().map(|l| &**l).collect();
            if let Err(e) = host.execute(&q, &refs, &[], &[]) {
                stats.refused += 1;
                tracing::warn!(error = %e, "a piece of title work was refused");
            }
            // The command buffer that uses these heaps now has its real serial;
            // entries may be repointed once it completes.
            let (now, _) = host.backend.queue_progress(&*q);
            for heap in heaps {
                heap.last_serial.set(now);
            }
        };
        let mut signal = |n: usize, _: u64| {
            let Some((queue, fence, value)) = signals.remove(&n) else {
                return;
            };
            if let (Some(q), Some(f)) = (queues.get(&queue), fences.get(&fence))
                && let Err(e) = host.execute(q, &[], &[], &[(&**f, value)])
            {
                tracing::warn!(error = %e, "a held fence signal was not queued");
            }
        };
        let done = held.pump(&mut World {
            read: &read,
            fence: &fence,
            run: &mut run,
            signal: &mut signal,
        });
        stats.waited += done.waited.len() as u64;
    }

    /// Queues whose head has been held longer than `limit`: a line each, with the waits that do
    /// not hold (address, comparison, reference, mask, current value).
    pub fn stuck(&self, limit: std::time::Duration) -> Vec<String> {
        let mut out: Vec<String> = self
            .requested
            .iter()
            .filter_map(|(k, &(want, since))| {
                let f = self.fences.get(k)?;
                let done = self.host.backend.completed(f);
                (done < want && since.elapsed() >= limit)
                    .then(|| format!("fence {k:#x}: signal of {want} queued {:.1?} ago, GPU at {done}", since.elapsed()))
            })
            .collect();
        let memory = self.host.memory();
        let read = |address: u64, bytes: u8| read_title(&memory, address, bytes);
        out.extend(self
            .held
            .blocked(&read)
            .into_iter()
            .filter(|(.., since, _)| *since >= limit)
            .map(|(q, items, since, unmet)| {
                let waits: Vec<String> = unmet
                    .iter()
                    .map(|w| {
                        format!(
                            "{:#x} cmp {} ref {:#x} mask {:#x} now {:?}",
                            w.address,
                            w.comparison,
                            w.reference,
                            w.mask,
                            read(w.address, w.bytes)
                        )
                    })
                    .collect();
                format!("queue {q:#x}: {items} held, head for {since:.1?}, waits {waits:?}")
            }));
        for (&id, queue) in &self.queues {
            let (submitted, completed) = self.host.backend.queue_progress(&**queue);
            if submitted > completed {
                out.push(format!(
                    "metal queue {id:#x}: {submitted} submitted, {completed} completed"
                ));
            }
        }
        out
    }

    /// Gives the host the app's drawable layer (`CAMetalLayer` on Metal, by address): frames the
    /// title presents are shown there from now on.
    pub fn set_surface(&mut self, layer: *mut std::ffi::c_void) -> Result<(), String> {
        self.surface = Some(self.host.backend.create_surface(layer)?);
        Ok(())
    }

    /// `PresentX`: title texture `key` shown over the surface, after the work queue `queue` has
    /// been given. shortcut: not behind work the queue still holds back (the frame is shown as
    /// far as it ran); `Ok(false)` when there is no surface or the title has no such texture.
    pub fn present(&mut self, queue: usize, key: usize) -> Result<bool, String> {
        self.pump();
        if std::env::var_os("VICTORY_TRACE_TEXTURE_MEMORY").is_some() {
            let frame = crate::frame::current();
            if frame.is_multiple_of(64) {
                let (n, bytes, linear_n, linear_bytes) = self.resident();
                let allocated = self.allocated();
                let key_bytes = self.key_bytes();
                let (pools, pool_bytes, mappings, mapped_bytes) = self.host.pool_stats();
                #[cfg(target_os = "macos")]
                let (footprint, swap) = (
                    victory_platform::mem::footprint(),
                    victory_platform::mem::swap_used(),
                );
                #[cfg(not(target_os = "macos"))]
                let (footprint, swap) = (0usize, 0usize);
                let usage = self.host.backend.memory_usage();
                tracing::info!(
                    frame,
                    textures = n,
                    bytes,
                    allocated,
                    key_bytes,
                    aliased = key_bytes.saturating_sub(bytes),
                    linear_n,
                    linear_bytes,
                    unreferenced = self.unreferenced(),
                    unset_roots = self.host.unset_roots(),
                    no_targets = self.host.no_targets(),
                    passes = self.host.passes(),
                    realize_whole = self.stats.realize_whole,
                    realize_ranged = self.stats.realize_ranged,
                    realize_required = self.stats.realize_required,
                    realize_conservative = self.stats.realize_conservative,
                    realize_deferred = self.stats.realize_deferred,
                    "host textures resident"
                );
                tracing::info!(
                    frame,
                    pools,
                    pool_bytes,
                    guest_resident = self.host.guest_resident(),
                    mappings,
                    mapped_bytes,
                    descriptor_heaps = self.host.descriptor_heap_bytes(),
                    footprint,
                    swap,
                    metal_allocated = usage.allocated,
                    metal_title_buffers = usage.title_buffers,
                    metal_host = usage.allocated.saturating_sub(usage.title_buffers),
                    metal_textures = allocated as u64,
                    metal_other = usage
                        .allocated
                        .saturating_sub(usage.title_buffers)
                        .saturating_sub(allocated as u64),
                    metal_recommended = usage.recommended,
                    metal_headroom = usage.recommended.saturating_sub(usage.allocated),
                    "host memory"
                );
                let (kinds, top) = self.breakdown();
                for age in [60u64, 600] {
                    let (heaps, entries, keys) = self.host.stale_heaps(age);
                    let bytes = self.stale_bytes(&keys);
                    tracing::info!(
                        frame,
                        age,
                        heaps,
                        entries,
                        keys = keys.len(),
                        bytes,
                        "idle descriptor heaps"
                    );
                }
                for (kind, keys, kind_bytes) in kinds {
                    tracing::info!(frame, kind, keys, bytes = kind_bytes, "host textures by kind");
                }
                if frame.is_multiple_of(512) {
                    let (groups, extra_keys, extra_bytes, overlap_bytes) =
                        self.alias_opportunities();
                    tracing::info!(
                        frame,
                        groups,
                        extra_keys,
                        extra_bytes,
                        overlap_bytes,
                        "host texture alias opportunities"
                    );
                    for (i, t) in top.iter().enumerate() {
                        tracing::info!(frame, rank = i, texture = t.as_str(), "largest host textures");
                    }
                }
            }
        }
        if self.surface.is_none() {
            return Ok(false);
        }
        let Some(tex) = self.texture_of(key) else {
            return Ok(false);
        };
        let q = self.queue(queue)?;
        if let Some(surface) = &self.surface {
            self.host.backend.present(&q, surface, &tex)?;
        }
        Ok(true)
    }

    /// The surface presented to (to read the frame counter, say).
    pub fn surface(&self) -> Option<&B::Surface> {
        self.surface.as_ref()
    }

    /// Whether any queue holds work (a timer should test it again soon).
    pub fn holds(&self) -> bool {
        !self.held.is_empty()
    }

    /// Makes sure title queue `key` has a backend queue before work is held for it.
    pub fn ensure_queue(&mut self, key: usize) -> Result<(), String> {
        self.queue(key).map(|_| ())
    }
}

/// The value a wait tests: from the pool when the address lies in one, else read from the
/// process's own memory (which may fault-free fail: the read is checked by the kernel).
fn read_title(memory: &PoolTable, address: u64, bytes: u8) -> Option<u64> {
    let n = usize::from(bytes.min(8));
    let at = match memory.resolve(address, n as u64) {
        Some((_, p)) => p,
        None => address as *const u8,
    };
    let mut v = [0u8; 8];
    crate::native::safe_read(at, &mut v[..n]).then(|| u64::from_le_bytes(v))
}
