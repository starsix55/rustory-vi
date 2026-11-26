//! `ComputeBackend` on Metal: `MTLCommandQueue`s, `MTLSharedEvent` fences, DXIL through the Metal
//! Shader Converter, dispatch with a top-level argument buffer at slot 2 over zero-copy memory.

use crate::backend::{ComputeBackend, MemoryUsage, RootLayout, Submission};
use crate::msc::Msc;
use crate::resources::GraphicsBackend;
use objc2::rc::Retained;
use objc2::runtime::{NSObjectProtocol, ProtocolObject};
use objc2_foundation::{NSString, NSURL};
use objc2_metal::*;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

type Dev = Retained<ProtocolObject<dyn MTLDevice>>;

pub struct Metal {
    pub(crate) dev: Dev,
    pub(crate) msc: Msc,
    /// Heap residency by `MTLResidencySet` (macOS 15+); off = `useResource` per encoder.
    pub(crate) residency_sets: bool,
    /// The set of the title's pool buffers, made when the first one is.
    pool_set: std::cell::RefCell<Option<Retained<ProtocolObject<dyn MTLResidencySet>>>>,
    /// What the encoders did to make buffers resident, for the tests and the frame line.
    pub(crate) counts: std::cell::Cell<ResidencyCounts>,
}

/// Counters of residency work.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct ResidencyCounts {
    /// `useResource` calls on title buffers (compute, render and blit encoders).
    pub use_resource: u64,
    /// Buffers added to the pool set.
    pub added: u64,
    /// Queues the pool set was added to.
    pub queues: u64,
}

impl Metal {
    /// Whether residency sets are in use (macOS 15 and later, unless switched off).
    pub fn residency_sets(&self) -> bool {
        self.residency_sets
    }

    pub fn residency_counts(&self) -> ResidencyCounts {
        self.counts.get()
    }

    pub(crate) fn count(&self, f: impl FnOnce(&mut ResidencyCounts)) {
        let mut c = self.counts.get();
        f(&mut c);
        self.counts.set(c);
    }
}

/// Library file names are unique per process, not per backend: Metal keeps libraries by URL.
static LIBRARIES: AtomicU64 = AtomicU64::new(0);

pub struct Queue(
    pub(crate) Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// Whether the pool residency set was added to it.
    pub(crate) std::cell::Cell<bool>,
    /// With the set, encoders no longer name the buffers they reach by address, and Metal's hazard
    /// tracking cannot see those accesses: command buffers wait for the one before through this
    /// event (and encoders within one through a fence), counted by `.3`.
    pub(crate) Retained<ProtocolObject<dyn MTLEvent>>,
    pub(crate) std::cell::Cell<u64>,
);
pub struct Fence(pub(crate) Retained<ProtocolObject<dyn MTLSharedEvent>>);
pub struct Pipeline(pub(crate) Retained<ProtocolObject<dyn MTLComputePipelineState>>);

/// A wrapped range of title memory.
pub struct Buffer {
    pub(crate) buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) base: *const u8,
    /// The residency set it is in (so that dropping it leaves the set).
    pub(crate) resident: std::cell::RefCell<Option<Retained<ProtocolObject<dyn MTLResidencySet>>>>,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if let Some(set) = self.resident.get_mut().take() {
            set.removeAllocation(ProtocolObject::from_ref(&*self.buf));
            set.commit();
        }
    }
}

impl Buffer {
    pub(crate) fn is_resident(&self) -> bool {
        self.resident.borrow().is_some()
    }
}

impl Metal {
    /// The system GPU and the installed converter (`VICTORY_MSC_DYLIB` overrides its path).
    pub fn new() -> Result<Metal, String> {
        let dev = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let residency_sets = std::env::var_os("VICTORY_METAL_NO_RESIDENCY_SETS").is_none()
            && dev.respondsToSelector(objc2::sel!(newResidencySetWithDescriptor:error:));
        Ok(Metal {
            dev,
            msc: Msc::open_default(),
            residency_sets,
            pool_set: Default::default(),
            counts: Default::default(),
        })
    }

    fn compute_from(
        &self,
        converted: Result<(i32, Vec<u8>), crate::msc::ConvError>,
    ) -> Result<Pipeline, String> {
        let (_, lib) = converted.map_err(|e| e.to_string())?;
        let l = self.load_library(&lib)?;
        let name = l
            .functionNames()
            .iter()
            .next()
            .ok_or("converted library is empty")?;
        let f = l.newFunctionWithName(&name).ok_or("no function")?;
        let p = self
            .dev
            .newComputePipelineStateWithFunction_error(&f)
            .map_err(|e| e.localizedDescription().to_string())?;
        Ok(Pipeline(p))
    }

    /// Heap residency through `MTLResidencySet` (`true`) or `useResource` in every encoder (`false`).
    pub fn with_residency_sets(mut self, on: bool) -> Metal {
        self.residency_sets = on;
        self
    }

    /// A metallib image to a library. shortcut: it goes through a file (no dispatch_data here); one
    /// name per library, because Metal keeps libraries by URL. Use `newLibraryWithData` once dispatch2
    /// is a dependency.
    pub(crate) fn load_library(
        &self,
        lib: &[u8],
    ) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, String> {
        let n = LIBRARIES.fetch_add(1, Ordering::Relaxed);
        let file =
            std::env::temp_dir().join(format!("victory-metal-{}-{n}.metallib", std::process::id()));
        std::fs::write(&file, lib).map_err(|e| e.to_string())?;
        let url = NSURL::fileURLWithPath(&NSString::from_str(file.to_str().unwrap()));
        let l = self.dev.newLibraryWithURL_error(&url);
        let _ = std::fs::remove_file(&file);
        l.map_err(|e| e.localizedDescription().to_string())
    }
}

impl ComputeBackend for Metal {
    type Queue = Queue;
    type Fence = Fence;
    type Buffer = Buffer;
    type Pipeline = Pipeline;
    type Heap = crate::gfx::Heap;

    fn memory_usage(&self) -> MemoryUsage {
        MemoryUsage {
            allocated: self.dev.currentAllocatedSize() as u64,
            recommended: self.dev.recommendedMaxWorkingSetSize(),
        }
    }

    fn create_queue(&self) -> Result<Queue, String> {
        let q = self.dev.newCommandQueue().ok_or("newCommandQueue failed")?;
        let order = self.dev.newEvent().ok_or("newEvent failed")?;
        Ok(Queue(q, std::cell::Cell::new(false), order, std::cell::Cell::new(0)))
    }

    fn create_fence(&self, initial: u64) -> Result<Fence, String> {
        let e = self.dev.newSharedEvent().ok_or("newSharedEvent failed")?;
        e.setSignaledValue(initial);
        Ok(Fence(e))
    }

    fn wrap_title_memory(&self, ptr: *mut u8, len: usize) -> Result<Buffer, String> {
        let p = NonNull::new(ptr.cast()).ok_or("null title memory")?;
        // SAFETY: the caller keeps `ptr..ptr+len` mapped while the buffer and its work live.
        let buf = unsafe {
            self.dev
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    p,
                    len,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
        }
        .ok_or("newBufferWithBytesNoCopy refused the range (alignment or size)")?;
        Ok(Buffer {
            buf,
            base: ptr,
            resident: Default::default(),
        })
    }

    fn make_resident(&self, queue: &Queue, buffers: &[&Buffer]) -> Result<(), String> {
        if !self.residency_sets {
            return Ok(());
        }
        let set = {
            let mut slot = self.pool_set.borrow_mut();
            match &*slot {
                Some(s) => s.clone(),
                None => {
                    let s = self
                        .dev
                        .newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new())
                        .map_err(|e| e.localizedDescription().to_string())?;
                    *slot = Some(s.clone());
                    s
                }
            }
        };
        let mut joined = false;
        for b in buffers.iter().filter(|b| !b.is_resident()) {
            set.addAllocation(ProtocolObject::from_ref(&*b.buf));
            *b.resident.borrow_mut() = Some(set.clone());
            self.count(|c| c.added += 1);
            joined = true;
        }
        if joined {
            set.commit();
        }
        if !queue.1.replace(true) {
            queue.0.addResidencySet(&set);
            self.count(|c| c.queues += 1);
        }
        Ok(())
    }

    fn gpu_address(&self, b: &Buffer, title: *const u8) -> u64 {
        b.buf.gpuAddress() + (title as usize - b.base as usize) as u64
    }

    fn create_compute_pipeline(&self, dxil: &[u8]) -> Result<Pipeline, String> {
        self.compute_from(self.msc.convert(dxil, None))
    }

    fn create_compute_pipeline_with(
        &self,
        dxil: &[u8],
        layout: &RootLayout,
    ) -> Result<Pipeline, String> {
        self.compute_from(self.msc.convert_with(dxil, None, Some(layout)))
    }

    fn submit(&self, queue: &Queue, work: &Submission<'_, Self>) -> Result<(), String> {
        use crate::resources::{Pass, PassSubmission};
        self.submit_passes(
            queue,
            &PassSubmission {
                waits: work.waits,
                passes: &[Pass::Compute(work.dispatches)],
                signals: work.signals,
            },
        )
    }

    fn signal(&self, fence: &Fence, value: u64) {
        fence.0.setSignaledValue(value);
    }

    fn completed(&self, fence: &Fence) -> u64 {
        fence.0.signaledValue()
    }

    fn wait(&self, fence: &Fence, value: u64, timeout_ms: u64) -> bool {
        fence.0.waitUntilSignaledValue_timeoutMS(value, timeout_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Dispatch, RootArgs};
    use crate::mem::{Region, page_size};

    /// `dst[i] = src[i] * k.x + k.y` for `numthreads(64)`, root signature
    /// `RootConstants(num32BitConstants=2, b0), SRV(t0), UAV(u0)` (the `--demo` shader, compiled by dxc;
    /// our own HLSL, not a title shader).
    const AFFINE: &[u8] = include_bytes!("../tests/affine.dxil");
    const N: usize = 4096;

    fn backend() -> Option<Metal> {
        if !std::path::Path::new(crate::msc::DEFAULT_DYLIB).exists()
            && std::env::var_os("VICTORY_MSC_DYLIB").is_none()
        {
            eprintln!("skipped: Metal Shader Converter not installed");
            return None;
        }
        Some(Metal::new().unwrap())
    }

    /// Three arrays in one zero-copy buffer: a, b, c.
    struct Mem {
        region: Region,
        buf: Buffer,
        part: usize,
    }

    impl Mem {
        fn new(m: &Metal) -> Mem {
            let part = (N * 4).next_multiple_of(page_size());
            let region = Region::new(3 * part, None).unwrap();
            let buf = m.wrap_title_memory(region.ptr, region.len).unwrap();
            Mem { region, buf, part }
        }
        fn at(&self, i: usize) -> *mut u32 {
            // SAFETY: i < 3, inside the region.
            unsafe { self.region.ptr.add(i * self.part).cast() }
        }
        fn gpu(&self, m: &Metal, i: usize) -> u64 {
            m.gpu_address(&self.buf, self.at(i).cast())
        }
        fn set(&self, i: usize, f: impl Fn(usize) -> u32) {
            // SAFETY: the GPU is idle or not yet submitted when tests call this.
            (0..N).for_each(|j| unsafe { self.at(i).add(j).write_volatile(f(j)) });
        }
        fn get(&self, i: usize, j: usize) -> u32 {
            // SAFETY: tests read after the fence wait.
            unsafe { self.at(i).add(j).read_volatile() }
        }
    }

    /// `out = in * mul + add` over a root signature of constants, SRV, UAV.
    fn step<'a>(
        m: &Metal,
        p: &'a Pipeline,
        mem: &'a Mem,
        (src, dst): (usize, usize),
        (mul, add): (u32, u32),
    ) -> Dispatch<'a, Metal> {
        Dispatch {
            pipeline: p,
            root: RootArgs::default()
                .constants(&[mul, add])
                .address(mem.gpu(m, src))
                .address(mem.gpu(m, dst)),
            resources: vec![&mem.buf],
            heaps: Default::default(),
            groups: [N / 64, 1, 1],
            threads_per_group: [64, 1, 1],
        }
    }

    /// The D3D12 pattern of the title: queue A dispatches and signals fence N; queue B waits for N
    /// and dispatches over the same memory; the CPU waits. B is submitted first, so it is the fence
    /// and not the submission order that makes the result right.
    #[test]
    fn queue_b_waits_for_queue_a_over_the_same_buffer() {
        let Some(m) = backend() else { return };
        let (qa, qb) = (m.create_queue().unwrap(), m.create_queue().unwrap());
        let fence = m.create_fence(0).unwrap();
        let done = m.create_fence(0).unwrap();
        let p = m.create_compute_pipeline(AFFINE).unwrap();
        let mem = Mem::new(&m);
        mem.set(0, |i| i as u32);

        // B: c = b * 3, after fence reaches 7; signals `done` 1.
        let b_work = [step(&m, &p, &mem, (1, 2), (3, 0))];
        m.submit(
            &qb,
            &Submission {
                waits: &[(&fence, 7)],
                dispatches: &b_work,
                signals: &[(&done, 1)],
            },
        )
        .unwrap();
        assert!(!m.wait(&done, 1, 50), "B ran before A signalled");
        assert_eq!(m.completed(&fence), 0);

        // A: b = a * 2 + 1; signals fence 7.
        let a_work = [step(&m, &p, &mem, (0, 1), (2, 1))];
        m.submit(
            &qa,
            &Submission {
                waits: &[],
                dispatches: &a_work,
                signals: &[(&fence, 7)],
            },
        )
        .unwrap();
        assert!(m.wait(&done, 1, 5000), "timed out");
        assert!(m.completed(&fence) >= 7);
        let ok = (0..N).all(|i| mem.get(2, i) == (i as u32 * 2 + 1) * 3);
        assert!(ok, "c[i] must be (a[i]*2+1)*3");
    }

    /// `ID3D12Fence::Signal` from the CPU releases a queue that waits (a title Signal / `hold.rs`).
    #[test]
    fn cpu_signal_releases_a_waiting_queue() {
        let Some(m) = backend() else { return };
        let q = m.create_queue().unwrap();
        let (gate, done) = (m.create_fence(0).unwrap(), m.create_fence(0).unwrap());
        let p = m.create_compute_pipeline(AFFINE).unwrap();
        let mem = Mem::new(&m);
        mem.set(0, |i| i as u32);
        let w = [step(&m, &p, &mem, (0, 1), (5, 7))];
        m.submit(
            &q,
            &Submission {
                waits: &[(&gate, 1)],
                dispatches: &w,
                signals: &[(&done, 1)],
            },
        )
        .unwrap();
        assert!(!m.wait(&done, 1, 50));
        m.signal(&gate, 1);
        assert!(m.wait(&done, 1, 5000));
        assert!((0..N).all(|i| mem.get(1, i) == i as u32 * 5 + 7));
    }

    /// Many submissions in flight over several queues, chained by fence values, one CPU wait: a
    /// frame's worth of the pattern above (35 submissions, alternating queues, `x = x * 1 + 1`).
    #[test]
    fn many_submissions_in_flight_one_wait() {
        let Some(m) = backend() else { return };
        const SUBS: u64 = 64;
        let queues: Vec<_> = (0..4).map(|_| m.create_queue().unwrap()).collect();
        let chain = m.create_fence(0).unwrap();
        let p = m.create_compute_pipeline(AFFINE).unwrap();
        let mem = Mem::new(&m);
        mem.set(0, |_| 0);
        // Ping-pong between arrays 0 and 1: each step adds 1 to what the previous step wrote.
        let works: Vec<_> = (0..SUBS)
            .map(|s| {
                let (src, dst) = if s % 2 == 0 { (0, 1) } else { (1, 0) };
                [step(&m, &p, &mem, (src, dst), (1, 1))]
            })
            .collect();
        let t = std::time::Instant::now();
        // In order (a queue runs its command buffers in order, so reversing would deadlock): only the
        // fence makes queue k+1 wait for queue k.
        for s in 0..SUBS {
            let waits = [(&chain, s)];
            let signals = [(&chain, s + 1)];
            m.submit(
                &queues[s as usize % queues.len()],
                &Submission {
                    waits: &waits,
                    dispatches: &works[s as usize],
                    signals: &signals,
                },
            )
            .unwrap();
        }
        assert!(
            m.wait(&chain, SUBS, 10_000),
            "timed out at {}",
            m.completed(&chain)
        );
        eprintln!(
            "{SUBS} dependent submissions over 4 queues: {:?}",
            t.elapsed()
        );
        // SUBS is even: the last write is array 0.
        assert!((0..N).all(|i| mem.get(0, i) == SUBS as u32));
    }
}
