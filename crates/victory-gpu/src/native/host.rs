//! The native host: what `host.rs` is for D3D12, over any [`GraphicsBackend`].
//!
//! One thread drives it (backends hold Objective-C objects). Queues, fences, pipelines, descriptor
//! heaps and textures are made here; title lists are [`NativeList`]s; [`NativeHost::execute`] turns
//! lists into passes and submits them without waiting. Title memory is wrapped lazily, one buffer
//! per pool (see [`super::memory`]); nothing is copied to or from it.

use super::addr::{self, Mapping, Pages, Pairs};
use super::descriptors::{Decoded, decode};
use super::list::{Binding, Cmd, ComputePso, NativeHeap, NativeList, Root, Targets};
use super::memory::PoolTable;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::sync::Arc;
use victory_backend::backend::{Dispatch, Heaps, RootArgs};
use victory_backend::resources::*;

/// A title resource object the host knows by its key (the object's address on the title's side).
enum Res<B: GraphicsBackend> {
    /// A buffer: title address and size.
    Buffer(u64, u64),
    Texture(Rc<B::Texture>),
}

pub struct NativeHost<B: GraphicsBackend> {
    pub backend: B,
    memory: Arc<PoolTable>,
    /// A buffer over each pool, made the first time the GPU needs it.
    buffers: RefCell<HashMap<usize, Rc<B::Buffer>>>,
    resources: RefCell<HashMap<usize, Res<B>>>,
    heaps: RefCell<Vec<Rc<NativeHeap<B>>>>,
    contract: RefCell<Contract<B>>,
    /// Title textures currently named by a descriptor heap entry. A heap entry
    /// retains the Metal texture itself, so these pin device memory even when
    /// the host's `Rc` map no longer holds the key.
    heap_refs: RefCell<HashMap<usize, usize>>,
    /// Draws and dispatches dropped because they read a root parameter the title never set.
    unset_roots: std::cell::Cell<u64>,
    /// Draws and clears dropped because none of their targets has a host texture.
    no_targets: std::cell::Cell<u64>,
    /// Passes handed to the backend.
    passes: std::cell::Cell<u64>,
}

/// How the error of a root parameter read with nothing set in it ends ([`NativeHost::execute`]
/// drops that draw or dispatch alone).
const NEVER_SET: &str = "was never set";

/// Drops one heap entry's reference to title texture `key`.
fn release_heap_ref(refs: &mut HashMap<usize, usize>, key: usize) {
    if let Some(n) = refs.get_mut(&key) {
        *n = n.saturating_sub(1);
        if *n == 0 {
            refs.remove(&key);
        }
    }
}

/// The host's side of the pool-offset contract ([`super::addr`]).
struct Contract<B: GraphicsBackend> {
    pairs: Pairs,
    /// Superseded tables, kept while submissions that read them may still run.
    // shortcut: the GPU is assumed not to be more than `RETIRED` table rebuilds behind.
    retired: VecDeque<Rc<Pages<B>>>,
    /// The memory the null view (pair 0) has, with size 0.
    null: Option<Rc<Pages<B>>>,
    /// A resource heap of the host's own, for shaders that need the pairs and the title bound none.
    heap: Option<Rc<NativeHeap<B>>>,
}

impl<B: GraphicsBackend> Default for Contract<B> {
    fn default() -> Self {
        Contract {
            pairs: Pairs::default(),
            retired: VecDeque::new(),
            null: None,
            heap: None,
        }
    }
}

const RETIRED: usize = 8;

/// What one command resolved to: title addresses are made into buffers here, kept alive until the
/// submission has been handed over.
struct Prep<B: GraphicsBackend> {
    root: RootArgs,
    /// The pixel stage's own arguments.
    ps_root: Option<RootArgs>,
    bufs: Vec<Rc<B::Buffer>>,
    vertex: Vec<(Rc<B::Buffer>, *const u8)>,
    index: Option<(Rc<B::Buffer>, *const u8)>,
    blit: Vec<(Rc<B::Buffer>, *const u8)>,
    texture: Option<TextureOp<B>>,
    /// The resource heap the work needs in place of the one the title bound (none bound).
    heap: Option<Rc<NativeHeap<B>>>,
}

/// A texture copy resolved from the title's resource objects at submission.
enum TextureOp<B: GraphicsBackend> {
    Copy {
        src: Rc<B::Texture>,
        dst: Rc<B::Texture>,
    },
    Region {
        src: Rc<B::Texture>,
        dst: Rc<B::Texture>,
        src_slice: u32,
        src_level: u32,
        src_origin: [u32; 3],
        size: [u32; 3],
        dst_slice: u32,
        dst_level: u32,
        dst_origin: [u32; 3],
    },
}

enum Group {
    Compute(usize, usize),
    Blit(usize, usize),
    Render(usize, usize),
    Clear(usize),
}

type HeapPair<B> = (Option<Rc<NativeHeap<B>>>, Option<Rc<NativeHeap<B>>>);

fn heaps<B: GraphicsBackend>(h: &HeapPair<B>) -> Heaps<'_, B> {
    Heaps {
        resource: h.0.as_ref().map(|h| &h.heap),
        sampler: h.1.as_ref().map(|h| &h.heap),
    }
}

/// The title's record in descriptor slot `handle` (tag, view index, generation).
fn record_at(handle: u64) -> [u8; 12] {
    let mut r = [0u8; 12];
    super::safe_read(handle as *const u8, &mut r);
    r
}

impl<B: GraphicsBackend> NativeHost<B> {
    pub fn new(backend: B) -> Self {
        Self::with_memory(backend, Arc::new(PoolTable::default()))
    }

    /// A host over a pool table the pool observer already feeds.
    pub fn with_memory(backend: B, memory: Arc<PoolTable>) -> Self {
        NativeHost {
            backend,
            memory,
            buffers: RefCell::default(),
            resources: RefCell::default(),
            heaps: RefCell::default(),
            contract: RefCell::default(),
            heap_refs: RefCell::default(),
            unset_roots: std::cell::Cell::new(0),
            no_targets: std::cell::Cell::new(0),
            passes: std::cell::Cell::new(0),
        }
    }

    /// Draws and clears dropped so far because none of their targets has a host texture.
    pub fn no_targets(&self) -> u64 {
        self.no_targets.get()
    }

    /// Draws and dispatches dropped so far because they read a root parameter never set.
    pub fn unset_roots(&self) -> u64 {
        self.unset_roots.get()
    }

    /// Passes handed to the backend so far (compute, render and blit; a clear is a render pass).
    pub fn passes(&self) -> u64 {
        self.passes.get()
    }

    /// The table the pool observer keeps current; register it as (or call it from) the observer.
    pub fn memory(&self) -> Arc<PoolTable> {
        self.memory.clone()
    }

    /// The host texture of title resource `key`, if the title realized it.
    fn texture_resource(&self, key: usize) -> Result<Rc<B::Texture>, String> {
        match self.resources.borrow().get(&key) {
            Some(Res::Texture(t)) => Ok(t.clone()),
            _ => Err(format!("texture resource {key:#x} is not realized")),
        }
    }

    /// The buffer and pool-mapping address of `[addr, addr + len)` of title memory.
    pub fn resolve(&self, addr: u64, len: u64) -> Result<(Rc<B::Buffer>, *const u8), String> {
        let mut buffers = self.buffers.borrow_mut();
        for gone in self.memory.take_released() {
            buffers.remove(&gone);
        }
        let (pool, ptr) = self
            .memory
            .resolve(addr, len)
            .ok_or_else(|| format!("title address {addr:#x}+{len:#x} is not in a mapped pool"))?;
        drop(buffers);
        Ok((self.pool_buffer(pool.id, pool.base, pool.len)?, ptr))
    }

    /// The buffer over pool `id`, made the first time the GPU needs it.
    fn pool_buffer(&self, id: usize, base: usize, len: usize) -> Result<Rc<B::Buffer>, String> {
        let mut buffers = self.buffers.borrow_mut();
        Ok(match buffers.entry(id) {
            std::collections::hash_map::Entry::Occupied(e) => e.get().clone(),
            std::collections::hash_map::Entry::Vacant(e) => e
                .insert(Rc::new(self.backend.wrap_title_memory(base as *mut u8, len)?))
                .clone(),
        })
    }

    pub fn compute_pipeline(
        &self,
        dxil: &[u8],
        threads: [usize; 3],
    ) -> Result<Rc<ComputePso<B>>, String> {
        Ok(Rc::new(ComputePso {
            pso: self.backend.create_compute_pipeline(dxil)?,
            threads,
            binding: None,
        }))
    }

    pub fn render_pipeline(
        &self,
        desc: &RenderPipelineDesc<'_>,
    ) -> Result<Rc<B::RenderPipeline>, String> {
        self.backend.create_render_pipeline(desc).map(Rc::new)
    }

    pub fn texture(&self, desc: &TextureDesc) -> Result<Rc<B::Texture>, String> {
        self.backend.create_texture(desc).map(Rc::new)
    }

    /// Names a buffer resource object `key` (what `CreateShaderResourceView` is passed) at title memory.
    pub fn register_buffer(&self, key: usize, addr: u64, len: u64) {
        self.resources
            .borrow_mut()
            .insert(key, Res::Buffer(addr, len));
    }

    pub fn register_texture(&self, key: usize, texture: &Rc<B::Texture>) {
        self.resources
            .borrow_mut()
            .insert(key, Res::Texture(texture.clone()));
    }

    /// A shader-visible heap over the title's descriptor memory at `start` (zeroed memory the title
    /// writes views into; 32 bytes an entry, 16 for samplers).
    pub fn heap(
        &self,
        kind: HeapKind,
        start: u64,
        count: usize,
    ) -> Result<Rc<NativeHeap<B>>, String> {
        // A resource heap is longer than the title made it by the entries that hold the pool-offset
        // contract's view pairs.
        let entries = if kind == HeapKind::Resource {
            if !addr::fits(count) {
                tracing::warn!(
                    count,
                    "a resource heap too long for title memory views: shaders that find memory by address will not run with it"
                );
            }
            count + addr::RESERVED
        } else {
            count
        };
        let heap = Rc::new(NativeHeap {
            heap: self.backend.create_heap(kind, entries)?,
            kind,
            start,
            count,
            pairs: Default::default(),
            table: Default::default(),
            mirrored: Default::default(),
            keys: Default::default(),
            last_used: Default::default(),
        });
        self.heaps.borrow_mut().push(heap.clone());
        Ok(heap)
    }

    /// The heap `handle` (a descriptor's address, a root table's) lies in, and the entry's index.
    pub fn heap_at(&self, handle: u64) -> Option<(Rc<NativeHeap<B>>, usize)> {
        self.heaps.borrow().iter().find_map(|h| {
            let index = (handle.checked_sub(h.start)? / Self::step(h.kind)) as usize;
            (index < h.count).then(|| (h.clone(), index))
        })
    }

    /// The title released the heap whose descriptors start at `start`.
    pub fn drop_heap(&self, start: u64) {
        let mut heaps = self.heaps.borrow_mut();
        let Some(at) = heaps.iter().position(|h| h.start == start) else {
            return;
        };
        let heap = heaps.remove(at);
        let keys: Vec<usize> = heap.keys.borrow().values().copied().collect();
        let mut refs = self.heap_refs.borrow_mut();
        for key in keys {
            release_heap_ref(&mut refs, key);
        }
    }

    /// Title textures named by a live descriptor heap entry, and how many
    /// entries name each. Snapshot for the residency report.
    pub fn heap_texture_refs(&self) -> HashMap<usize, usize> {
        self.heap_refs.borrow().clone()
    }

    /// Heaps no draw, dispatch, descriptor write or table check has named for
    /// `age` frames: `(heaps, entries, texture keys)`. A diagnostic: an idle
    /// heap may still be live and is not reclaimed by this call.
    pub fn stale_heaps(&self, age: u64) -> (usize, usize, Vec<usize>) {
        let now = crate::frame::current();
        let (mut heaps, mut entries, mut keys) = (0, 0, Vec::new());
        for h in self.heaps.borrow().iter() {
            if now.saturating_sub(h.last_used.get()) < age {
                continue;
            }
            heaps += 1;
            let k = h.keys.borrow();
            entries += k.len();
            keys.extend(k.values().copied());
        }
        (heaps, entries, keys)
    }

    /// Records that heap entry `index` now names title texture `key` (or no
    /// texture). The old entry's reference is released first.
    fn note_heap_key(&self, heap: &NativeHeap<B>, index: usize, key: Option<usize>) {
        let old = heap.keys.borrow_mut().remove(&index);
        let mut refs = self.heap_refs.borrow_mut();
        if let Some(old) = old {
            release_heap_ref(&mut refs, old);
        }
        if let Some(key) = key {
            *refs.entry(key).or_default() += 1;
            heap.keys.borrow_mut().insert(index, key);
        }
    }

    /// The title released resource object `key`: later views naming it decode to nothing.
    pub fn forget_resource(&self, key: usize) {
        self.resources.borrow_mut().remove(&key);
    }

    /// Whether resource object `key` is registered (as a buffer or a texture).
    pub fn knows(&self, key: usize) -> bool {
        self.resources.borrow().contains_key(&key)
    }

    /// Committed guest pools and mappings: `(pools, pool bytes, mappings,
    /// mapped bytes)`. Pools count once however many titles address map them.
    pub fn pool_stats(&self) -> (usize, usize, usize, usize) {
        self.memory.stats()
    }

    /// Bytes of descriptor-heap buffers the host allocated. These are host
    /// memory, not guest, and are part of `currentAllocatedSize`.
    pub fn descriptor_heap_bytes(&self) -> u64 {
        self.heaps
            .borrow()
            .iter()
            .map(|h| h.count as u64 * Self::step(h.kind))
            .sum()
    }

    fn step(kind: HeapKind) -> u64 {
        u64::from(crate::descriptor::increment(u32::from(kind == HeapKind::Sampler)).unwrap_or(32))
    }

    /// The title wrote a view at descriptor `handle` (the call `descriptor::mirror` makes on
    /// D3D12): decode its record and write the backend's entry. Stale or unsupported views leave
    /// the entry as it was; returns whether an entry was written.
    pub fn mirror(&self, handle: u64) -> Result<bool, String> {
        let found = self.heaps.borrow().iter().find_map(|h| {
            let at = handle.checked_sub(h.start)?;
            let index = (at / Self::step(h.kind)) as usize;
            (index < h.count).then(|| (h.clone(), index))
        });
        let Some((heap, index)) = found else {
            return Ok(false);
        };
        heap.last_used.set(crate::frame::current());
        let Some(view) = crate::descriptor::view(handle) else {
            return Ok(false);
        };
        let resources = self.resources.borrow();
        let buffer_of = |r: usize| match resources.get(&r) {
            Some(Res::Buffer(a, n)) => Some((*a, *n)),
            _ => None,
        };
        let Some(decoded) = decode(&view, &buffer_of) else {
            return Ok(false);
        };
        match decoded {
            Decoded::Buffer { addr, size } => {
                let (buf, ptr) = self.resolve(addr, u64::from(size))?;
                self.backend
                    .write_buffer(&heap.heap, index, &buf, ptr, size);
                self.note_heap_key(&heap, index, None);
            }
            Decoded::Image {
                resource,
                mips,
                format,
            } => {
                let Some(Res::Texture(tex)) = resources.get(&resource) else {
                    return Ok(false);
                };
                match (mips, format) {
                    (Some((first, count)), Some(format)) if count > 0 => {
                        let v = self.backend.create_texture_view(
                            tex,
                            &ViewDesc {
                                format,
                                mips: (first as usize, count as usize),
                                slices: (0, 1),
                            },
                        )?;
                        self.backend.write_texture(&heap.heap, index, &v, 0.0);
                    }
                    _ => self.backend.write_texture(&heap.heap, index, tex, 0.0),
                }
                self.note_heap_key(&heap, index, Some(resource));
            }
            Decoded::Sampler { desc, lod_bias } => {
                let s = self.backend.create_sampler(&desc)?;
                self.backend.write_sampler(&heap.heap, index, &s, lod_bias);
                self.note_heap_key(&heap, index, None);
            }
        }
        heap.mirrored.borrow_mut().insert(index, record_at(handle));
        Ok(true)
    }

    /// Minimum entries of each bound table checked at submission when a shader has no
    /// metadata for that table (or declares fewer entries).
    const TABLE_CHECK: usize = 16;

    /// Mirrors again the entries of each table `root` binds whose title record changed since
    /// they were last mirrored. The title copies records into its shader-visible heaps with plain
    /// stores (the content draws' constant buffers, from frame ~700), which no device call reports,
    /// so their host entries stayed empty and the shaders read zeros.
    fn remirror_tables(&self, root: &[Option<Root<B>>], binding: Option<&Binding>) {
        for (slot, r) in root.iter().enumerate() {
            let Some(Root::Table(h, at)) = r else { continue };
            h.last_used.set(crate::frame::current());
            let check = table_check_count(slot, binding, Self::TABLE_CHECK);
            for index in *at..(*at + check).min(h.count) {
                let handle = h.start + index as u64 * Self::step(h.kind);
                let record = record_at(handle);
                if h.mirrored.borrow().get(&index) == Some(&record) {
                    continue;
                }
                // shortcut: views of resources the host has not realized yet are not made here
                // (`glue::mirror` does that for device writes); such an entry stays as it was.
                if let Err(e) = self.mirror(handle) {
                    tracing::debug!(error = %e, handle = format!("{handle:#x}"), "a table entry was not mirrored at submission");
                }
                // Tried once per record, whatever the result.
                h.mirrored.borrow_mut().insert(index, record);
            }
        }
    }

    /// The arguments of a translated title shader: its user-data block, then its root views.
    fn translated_args(
        &self,
        root: &[Option<Root<B>>],
        b: &Binding,
        heap: Option<&Rc<NativeHeap<B>>>,
        p: &mut Prep<B>,
    ) -> Result<(), String> {
        use victory_shader::emit::{ROOT_SRV_VIEW, ROOT_UAV_VIEW, ROOT_VIEW_OFFSET};
        use victory_shader::RootType;
        let set = |slot: u8| {
            root.get(usize::from(slot))
                .and_then(|r| r.as_ref())
                .ok_or_else(|| format!("root parameter {slot} {NEVER_SET}"))
        };
        let mut args = std::mem::take(&mut p.root);
        if b.dwords > 0 {
            let mut values = vec![0u32; b.dwords];
            for u in &b.user_data {
                values[u.index as usize] = match (u.root, set(u.slot)?) {
                    (RootType::Constants, Root::Constants(c)) => {
                        c.get(u.dword as usize).copied().unwrap_or(0)
                    }
                    // The byte offset of the table's first entry in its heap.
                    (RootType::Table, Root::Table(h, at)) => (*at as u64 * Self::step(h.kind)) as u32,
                    (RootType::Descriptor, Root::View(a)) if u.dword < 2 => {
                        (a >> (32 * u.dword)) as u32
                    }
                    // A view of the pool holding the root view, and the address's offset in it.
                    (RootType::Descriptor, Root::View(a))
                        if matches!(u.dword, ROOT_SRV_VIEW | ROOT_UAV_VIEW | ROOT_VIEW_OFFSET) =>
                    {
                        let heap = match &p.heap {
                            Some(h) => h.clone(),
                            None => match heap {
                                Some(h) => h.clone(),
                                None => self.host_heap()?,
                            },
                        };
                        p.heap = Some(heap.clone());
                        let (view, offset) = self.root_view(&heap, *a);
                        match u.dword {
                            ROOT_SRV_VIEW => view,
                            ROOT_UAV_VIEW => view + 1,
                            _ => offset,
                        }
                    }
                    (kind, _) => {
                        return Err(format!(
                            "root parameter {} is not what the shader reads it as ({kind:?}, dword {})",
                            u.slot, u.dword
                        ));
                    }
                };
            }
            args = args.constants(&values);
        }
        for &(slot, _) in &b.views {
            let Root::View(a) = set(slot)? else {
                return Err(format!("root parameter {slot} is not a root view"));
            };
            let (buf, ptr) = self.resolve(*a, 1)?;
            let gpu = self.backend.gpu_address(&buf, ptr);
            p.bufs.push(buf);
            args = args.address(gpu);
        }
        p.root = args;
        Ok(())
    }

    /// The page table of the pool-offset contract for `heap` and the pool table as it is now:
    /// rebuilt, into a buffer of its own, when a pool or a mapping changed since it was made.
    fn memory_table(&self, heap: &NativeHeap<B>) -> Result<Rc<Pages<B>>, String> {
        if !addr::fits(heap.count) {
            return Err(format!(
                "a resource heap of {} entries has no room for title memory views",
                heap.count
            ));
        }
        let (generation, mappings) = self.memory.mappings();
        if let Some((g, t)) = &*heap.table.borrow()
            && *g == generation
        {
            return Ok(t.clone());
        }
        let maps: Vec<Mapping> = mappings
            .iter()
            .map(|(at, len, pool, offset)| Mapping {
                at: *at as u64,
                len: *len as u64,
                pool: pool.id,
                base: pool.base,
                pool_len: pool.len as u64,
                offset: *offset as u64,
            })
            .collect();
        let mut c = self.contract.borrow_mut();
        let words = addr::build(&maps, &mut c.pairs, heap.count as u32);
        let table = Rc::new(Pages::new(&self.backend, words.len() * 4)?);
        table.write(&words);
        if let Some((_, old)) = heap.table.borrow_mut().replace((generation, table.clone())) {
            c.retired.push_back(old);
            while c.retired.len() > RETIRED {
                c.retired.pop_front();
            }
        }
        Ok(table)
    }

    /// Writes the view pairs the table names into `heap`'s reserved entries (those it lacks).
    fn write_pairs(&self, heap: &NativeHeap<B>) -> Result<(), String> {
        let mut c = self.contract.borrow_mut();
        let have = heap.pairs.get();
        let want = c.pairs.keys.len() + 1;
        for n in have..want {
            let index = addr::pair_index(heap.count as u32, n as u32) as usize;
            let (buffer, ptr, size) = if n == 0 {
                if c.null.is_none() {
                    c.null = Some(Rc::new(Pages::new(&self.backend, 1)?));
                }
                let null = c.null.as_ref().expect("made above");
                (null.buf.clone(), null.ptr().cast_const(), 0)
            } else {
                let key = c.pairs.keys[n - 1];
                let buffer = self.pool_buffer(key.pool, key.base, key.len as usize)?;
                let (from, size) = Pairs::span(&key);
                (buffer, (key.base + from as usize) as *const u8, size)
            };
            // Both of the pair name the same bytes (the converter's descriptors are the same).
            for i in [index, index + 1] {
                self.backend.write_buffer(&heap.heap, i, &buffer, ptr, size);
            }
        }
        heap.pairs.set(want);
        Ok(())
    }

    /// The pair view and the offset in it of root view address `a` (see [`addr::root_view`]).
    fn root_view(&self, heap: &NativeHeap<B>, a: u64) -> (u32, u32) {
        let maps = self.mapping_list();
        addr::root_view(&maps, &mut self.contract.borrow_mut().pairs, heap.count as u32, a)
    }

    /// The pool table as the page table builder reads it.
    fn mapping_list(&self) -> Vec<Mapping> {
        self.memory
            .mappings()
            .1
            .iter()
            .map(|(at, len, pool, offset)| Mapping {
                at: *at as u64,
                len: *len as u64,
                pool: pool.id,
                base: pool.base,
                pool_len: pool.len as u64,
                offset: *offset as u64,
            })
            .collect()
    }

    /// What a shader that finds title memory through the page table is given besides its user data:
    /// the table (the last root parameter) and, in the heap it reads, the view pairs.
    fn memory_args(
        &self,
        heap: Option<&Rc<NativeHeap<B>>>,
        p: &mut Prep<B>,
    ) -> Result<(), String> {
        let heap = match heap {
            Some(h) => h.clone(),
            None => self.host_heap()?,
        };
        let table = self.memory_table(&heap)?;
        self.write_pairs(&heap)?;
        let gpu = self.backend.gpu_address(&table.buf, table.ptr());
        p.bufs.push(table.buf.clone());
        p.root = std::mem::take(&mut p.root).address(gpu);
        p.heap = Some(heap);
        Ok(())
    }

    /// A resource heap of the host's own, for work that needs the view pairs and has no heap.
    fn host_heap(&self) -> Result<Rc<NativeHeap<B>>, String> {
        if let Some(h) = &self.contract.borrow().heap {
            return Ok(h.clone());
        }
        let heap = Rc::new(NativeHeap {
            heap: self.backend.create_heap(HeapKind::Resource, addr::RESERVED)?,
            kind: HeapKind::Resource,
            start: 0,
            count: 0,
            pairs: Default::default(),
            table: Default::default(),
            mirrored: Default::default(),
            keys: Default::default(),
            last_used: Default::default(),
        });
        self.contract.borrow_mut().heap = Some(heap.clone());
        Ok(heap)
    }

    fn root_args(&self, root: &[Option<Root<B>>], p: &mut Prep<B>) -> Result<(), String> {
        for (i, r) in root.iter().enumerate() {
            let args = std::mem::take(&mut p.root);
            p.root = match r
                .as_ref()
                .ok_or_else(|| format!("root parameter {i} {NEVER_SET}"))?
            {
                Root::Constants(w) => args.constants(w),
                Root::View(a) => {
                    let (buf, ptr) = self.resolve(*a, 1)?;
                    let gpu = self.backend.gpu_address(&buf, ptr);
                    p.bufs.push(buf);
                    args.address(gpu)
                }
                Root::Table(h, at) => args.address(self.backend.heap_address(&h.heap, *at)),
            };
        }
        Ok(())
    }

    /// `VICTORY_TRACE_TARGETS=1`: the root parameters of the first draws that have vertex buffers,
    /// and the title memory they name (descriptor words, then the first 64 bytes at the address a
    /// constant buffer descriptor holds).
    fn trace_draw(&self, root: &[Option<Root<B>>], binding: Option<&Binding>, dispatch: bool) {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: [AtomicU32; 2] = [AtomicU32::new(0), AtomicU32::new(0)];
        if std::env::var_os("VICTORY_TRACE_TARGETS").is_none() || N[usize::from(dispatch)].fetch_add(1, Ordering::Relaxed) >= 6 {
            return;
        }
        let read = |at: u64, n: usize| {
            let mut b = vec![0u8; n];
            let ok = super::safe_read(at as *const u8, &mut b);
            let words: Vec<String> = b.chunks(4).map(|w| format!("{:08x}", u32::from_le_bytes(w.try_into().unwrap()))).collect();
            format!("{}{}", if ok { "" } else { "UNREADABLE " }, words.join(" "))
        };
        for (i, r) in root.iter().enumerate() {
            match r {
                Some(Root::Constants(w)) => tracing::info!(slot = i, ?w, "draw root constants"),
                Some(Root::View(a)) => tracing::info!(slot = i, addr = format!("{a:#x}"), mem = read(*a, 64), "draw root view"),
                Some(Root::Table(h, at)) => {
                    let base = h.start + *at as u64 * Self::step(h.kind);
                    let d = read(base, 32 * 4);
                    tracing::info!(slot = i, heap_start = format!("{:#x}", h.start), index = at, kind = ?h.kind, descriptors = d, "draw root table");
                    for k in 0..4u64 {
                        let handle = base + k * Self::step(h.kind);
                        match crate::descriptor::view(handle) {
                            Some(crate::descriptor::View::Constant(a, n)) => tracing::info!(entry = k, addr = format!("{a:#x}"), size = n, mem = read(a, 64), "table constant buffer"),
                            Some(crate::descriptor::View::Shader(r, d)) => {
                                #[cfg(target_os = "macos")]
                                let info = crate::resource::texture_info(r as crate::object::This);
                                #[cfg(not(target_os = "macos"))]
                                let info: Option<super::textures::TextureInfo> = None;
                                let (nz, total) = info.as_ref().map_or((0, 0), |i| {
                                    let m = super::textures::read_memory(i.memory, i.size.min(1 << 22));
                                    (m.iter().filter(|&&b| b != 0).count(), m.len())
                                });
                                tracing::info!(entry = k, resource = format!("{r:#x}"), desc = ?d, texture = ?info.as_ref().map(|i| (i.format, i.width, i.height, i.memory as usize, i.size)), nonzero_bytes = nz, of = total, "table texture");
                            }
                            Some(crate::descriptor::View::Unordered(r, counter, d)) => {
                                let info = texture_info_of(r);
                                let buffer = self.resources.borrow().get(&r).map(|b| match b {
                                    Res::Buffer(a, n) => (*a, *n),
                                    _ => (0, 0),
                                });
                                tracing::info!(entry = k, resource = format!("{r:#x}"), counter, desc = ?d, texture = ?info.as_ref().map(|i| (i.format, i.width, i.height, i.memory as usize, i.size)), buffer = ?buffer, "table unordered view");
                            }
                            Some(crate::descriptor::View::Placed(_, res, d)) => tracing::info!(entry = k, ?res, ?d, "table placed view"),
                            Some(v) => tracing::info!(entry = k, ?v, "table view"),
                            None => {}
                        }
                    }
                }
                None => {}
            }
        }
        if let Some(b) = binding {
            tracing::info!(dwords = b.dwords, user_data = ?b.user_data, views = ?b.views, ps = ?b.ps.as_ref().map(|p| p.user_data.clone()), "draw binding");
        }
    }

    /// `VICTORY_TRACE_CONTENT=<dir>`: for the first draws into a depth target, every texture and
    /// constant buffer their descriptor tables hold: its view, the fraction of its title memory
    /// that is non-zero, and (8-bit colour textures) the title memory detiled and the host
    /// texture, written as `<dir>/c<n>_<entry>_{title,host}.ppm`.
    #[inline(never)]
    fn trace_content(&self, root: &[Option<Root<B>>], vertex: &[u64], call: &victory_backend::resources::DrawCall, binding: Option<&Binding>, targets: &Targets<B>) {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        static SEEN: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());
        static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        let start = START.get_or_init(std::time::Instant::now);
        let Some(v) = std::env::var_os("VICTORY_TRACE_CONTENT") else { return };
        // `<dir>[,<seconds>[,<format>]]`: from that many seconds after the first draw on; with a
        // format, only the draws into render targets of that title format, boot passes included.
        let v = v.to_string_lossy().into_owned();
        let mut parts = v.split(',');
        let dir = std::path::PathBuf::from(parts.next().unwrap_or_default());
        let after: u64 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        let only: Option<u32> = parts.next().and_then(|s| s.parse().ok());
        if start.elapsed().as_secs() < after {
            return;
        }
        let target = targets.color.first().and_then(|c| {
            self.resources.borrow().iter().find_map(|(k, r)| matches!(r, Res::Texture(t) if Rc::ptr_eq(t, c)).then_some(*k))
        });
        if let Some(f) = only
            && target.and_then(texture_info_of).map(|i| i.format) != Some(f)
        {
            return;
        }
        // The boot passes (a 3840-wide texture sampled, nothing else) are skipped.
        let boot = root.iter().any(|r| {
            let Some(Root::Table(h, at)) = r else { return false };
            let base = h.start + *at as u64 * Self::step(h.kind);
            (0..8u64).any(|k| match crate::descriptor::view(base + k * Self::step(h.kind)) {
                Some(crate::descriptor::View::Shader(r, _)) => texture_info_of(r).is_some_and(|i| i.width >= 3840),
                _ => false,
            })
        });
        // ... and the fullscreen triangle's (one 96-byte constant buffer and no texture).
        let plain = !root.iter().any(|r| {
            let Some(Root::Table(h, at)) = r else { return false };
            let base = h.start + *at as u64 * Self::step(h.kind);
            (0..8u64).any(|k| match crate::descriptor::view(base + k * Self::step(h.kind)) {
                Some(crate::descriptor::View::Shader(..) | crate::descriptor::View::Placed(..)) => true,
                Some(crate::descriptor::View::Constant(_, n)) => n != 96,
                _ => false,
            })
        });
        if only.is_none() && (boot || plain) {
            return;
        }
        let n = N.fetch_add(1, Ordering::Relaxed);
        if n >= 600 {
            return;
        }
        let ppm = |name: String, w: usize, h: usize, px: &[u8], bgra: bool| {
            let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
            out.extend(px.as_chunks::<4>().0.iter().flat_map(|p| if bgra { [p[2], p[1], p[0]] } else { [p[0], p[1], p[2]] }));
            let _ = std::fs::write(std::path::Path::new(&dir).join(name), out);
        };
        let kinds: Vec<String> = root
            .iter()
            .map(|r| match r {
                Some(Root::Constants(w)) => format!("constants{w:?}"),
                Some(Root::View(a)) => format!("view{a:#x}"),
                Some(Root::Table(_, at)) => format!("table@{at}"),
                None => "unset".into(),
            })
            .collect();
        let head = vertex.first().map(|&a| {
            let mut b = [0u8; 32];
            super::safe_read(a as *const u8, &mut b);
            b
        });
        let target = target.map(|k| format!("{k:#x}"));
        tracing::info!(draw = n, ?target, ?kinds, ?vertex, ?call, ?head, vs = ?binding.map(|b| (b.dwords, b.user_data.clone(), b.views.clone())), ps = ?binding.and_then(|b| b.ps.as_ref()).map(|p| (p.dwords, p.user_data.clone())), "content: draw");
        for (slot, r) in root.iter().enumerate() {
            let Some(Root::Table(h, at)) = r else { continue };
            let base = h.start + *at as u64 * Self::step(h.kind);
            for k in 0..8u64 {
                let handle = base + k * Self::step(h.kind);
                let (key, info, what) = match crate::descriptor::view(handle) {
                    Some(crate::descriptor::View::Constant(a, size)) => {
                        let mut b = vec![0u8; (size as usize).min(1 << 16)];
                        let ok = super::safe_read(a as *const u8, &mut b);
                        let nz = b.iter().filter(|&&x| x != 0).count();
                        let floats: Vec<f32> = b.as_chunks::<4>().0.iter().take(24).map(|w| f32::from_le_bytes(*w)).collect();
                        tracing::info!(draw = n, slot, entry = k, addr = format!("{a:#x}"), size, readable = ok, nonzero = nz, ?floats, "content: constant buffer");
                        continue;
                    }
                    Some(crate::descriptor::View::Shader(r, d)) => (r, texture_info_of(r), format!("srv {d:?}")),
                    Some(crate::descriptor::View::Placed(_, res, d)) if res[0] > 1 => {
                        let address = u64::from(d[10]) | u64::from(d[11]) << 32;
                        (super::textures::placed_key(address, &res), placed_info_of(address, &res), format!("placed {res:?}"))
                    }
                    _ => continue,
                };
                let Some(info) = info else {
                    tracing::info!(draw = n, slot, entry = k, what, "content: view without a texture");
                    continue;
                };
                let tiled = super::textures::read_memory(info.memory, info.size.min(1 << 26));
                let nz = tiled.iter().filter(|&&b| b != 0).count();
                tracing::info!(draw = n, slot, entry = k, key = format!("{key:#x}"), format = info.format, w = info.width, h = info.height, layers = info.depth_or_layers, flags = info.flags, at = format!("{:#x}", info.memory as usize), size = info.size, nonzero = nz, fraction = nz as f64 / tiled.len().max(1) as f64, what, "content: texture");
                // A single-channel 8-bit texture (A8, R8) is written as grey: the title's memory once,
                // the host texture at every draw (the last one standing).
                if matches!(info.format, 65 | 61) {
                    let (w, hh) = (info.width as usize, info.height as usize);
                    let grey = |px: &[u8]| px.iter().flat_map(|&v| [v, v, v, 255]).collect::<Vec<u8>>();
                    let mut px = vec![0u8; w * hh];
                    if !SEEN.lock().unwrap_or_else(|e| e.into_inner()).contains(&key) {
                        SEEN.lock().unwrap_or_else(|e| e.into_inner()).push(key);
                        if let Some(desc) = info.desc()
                            && let Ok(tmp) = self.backend.create_texture(&desc)
                            && super::textures::upload(&self.backend, &tmp, &info, &tiled).is_ok()
                        {
                            self.backend.read_texture(&tmp, 0, 0, &mut px, w, w * hh);
                            ppm(format!("{key:x}_title_{n}.ppm"), w, hh, &grey(&px), false);
                        }
                    }
                    if let Some(Res::Texture(t)) = self.resources.borrow().get(&key) {
                        self.backend.read_texture(t, 0, 0, &mut px, w, w * hh);
                        ppm(format!("{key:x}_host.ppm"), w, hh, &grey(&px), false);
                    }
                    continue;
                }
                if !matches!(info.format, 28 | 87 | 91 | 29) || SEEN.lock().unwrap_or_else(|e| e.into_inner()).contains(&key) {
                    continue;
                }
                SEEN.lock().unwrap_or_else(|e| e.into_inner()).push(key);
                let (w, hh) = (info.width as usize, info.height as usize);
                let bgra = matches!(info.format, 87 | 91);
                let mut px = vec![0u8; w * hh * 4];
                if let Some(desc) = info.desc()
                    && let Ok(tmp) = self.backend.create_texture(&desc)
                    && super::textures::upload(&self.backend, &tmp, &info, &tiled).is_ok()
                {
                    self.backend.read_texture(&tmp, 0, 0, &mut px, w * 4, w * hh * 4);
                    ppm(format!("c{n}_{k}_title.ppm"), w, hh, &px, bgra);
                }
                if let Some(Res::Texture(t)) = self.resources.borrow().get(&key) {
                    self.backend.read_texture(t, 0, 0, &mut px, w * 4, w * hh * 4);
                    ppm(format!("c{n}_{k}_host.ppm"), w, hh, &px, bgra);
                }
            }
        }
    }

    fn prep(&self, cmd: &Cmd<B>) -> Result<Prep<B>, String> {
        let mut p = Prep {
            root: RootArgs::default(),
            ps_root: None,
            bufs: vec![],
            vertex: vec![],
            index: None,
            blit: vec![],
            texture: None,
            heap: None,
        };
        match cmd {
            Cmd::Dispatch {
                root, pso, heaps, groups, ..
            } => {
                tracing::debug!(?groups, "dispatch");
                if std::env::var_os("VICTORY_TRACE_DISPATCHES").is_some() {
                    tracing::info!(?groups, threads = ?pso.threads, "dispatch");
                }
                if std::env::var_os("VICTORY_TRACE_TARGETS").is_some() {
                    static G: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    if G.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 6 {
                        tracing::info!(?groups, threads = ?pso.threads, "dispatch");
                    }
                }
                self.trace_draw(root, pso.binding.as_ref(), true);
                self.remirror_tables(root, pso.binding.as_ref());
                match &pso.binding {
                Some(b) => {
                    self.translated_args(root, b, heaps.0.as_ref(), &mut p)?;
                    if b.memory {
                        self.memory_args(heaps.0.as_ref(), &mut p)?;
                    } else if let Some(h) = p.heap.clone() {
                        self.write_pairs(&h)?;
                    }
                }
                None => self.root_args(root, &mut p)?,
            }},
            Cmd::Draw {
                root,
                binding,
                heaps,
                vertex,
                index,
                call,
                targets,
                ..
            } => {
                if !vertex.is_empty() {
                    self.trace_draw(root, binding.as_deref(), false);
                }
                self.remirror_tables(root, binding.as_deref());
                self.trace_content(root, vertex, call, binding.as_deref(), targets);
                match binding {
                    Some(b) => {
                        self.translated_args(root, b, heaps.0.as_ref(), &mut p)?;
                        if b.memory {
                            self.memory_args(heaps.0.as_ref(), &mut p)?;
                        }
                        if let Some(ps) = &b.ps {
                            let vs_args = std::mem::take(&mut p.root);
                            self.translated_args(root, ps, heaps.0.as_ref(), &mut p)?;
                            if ps.memory {
                                self.memory_args(heaps.0.as_ref(), &mut p)?;
                            }
                            p.ps_root = Some(std::mem::replace(&mut p.root, vs_args));
                        }
                        if !b.memory && !b.ps.as_ref().is_some_and(|ps| ps.memory)
                            && let Some(h) = p.heap.clone()
                        {
                            self.write_pairs(&h)?;
                        }
                    }
                    None => self.root_args(root, &mut p)?,
                }
                for a in vertex {
                    p.vertex.push(self.resolve(*a, 1)?);
                }
                if let Some((a, _)) = index {
                    p.index = Some(self.resolve(*a, 1)?);
                }
            }
            Cmd::Copy { dst, src, len } => {
                p.blit.push(self.resolve(*src, *len as u64)?);
                p.blit.push(self.resolve(*dst, *len as u64)?);
            }
            Cmd::Fill { dst, len, .. } => p.blit.push(self.resolve(*dst, *len as u64)?),
            Cmd::CopyTexture { dst, src } => {
                p.texture = Some(TextureOp::Copy {
                    src: self.texture_resource(*src)?,
                    dst: self.texture_resource(*dst)?,
                });
            }
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
            } => {
                p.texture = Some(TextureOp::Region {
                    src: self.texture_resource(*src)?,
                    dst: self.texture_resource(*dst)?,
                    src_slice: *src_slice,
                    src_level: *src_level,
                    src_origin: *src_origin,
                    size: *size,
                    dst_slice: *dst_slice,
                    dst_level: *dst_level,
                    dst_origin: *dst_origin,
                });
            }
            Cmd::Clear { .. } => {}
        }
        Ok(p)
    }

    /// `ExecuteCommandLists` plus the queue's waits and signals: resolve every title address, group
    /// the commands into compute, render and blit passes in order, and queue them. Never waits.
    pub fn execute(
        &self,
        queue: &B::Queue,
        lists: &[&NativeList<B>],
        waits: &[(&B::Fence, u64)],
        signals: &[(&B::Fence, u64)],
    ) -> Result<(), String> {
        let same = |a: &Cmd<B>, b: &Cmd<B>| match (a, b) {
            (Cmd::Dispatch { .. }, Cmd::Dispatch { .. }) => true,
            (Cmd::Copy { .. } | Cmd::Fill { .. }, Cmd::Copy { .. } | Cmd::Fill { .. }) => true,
            (Cmd::CopyTexture { .. }, Cmd::CopyTexture { .. }) => true,
            (Cmd::CopyTextureRegion { .. }, Cmd::CopyTextureRegion { .. }) => true,
            (Cmd::Draw { targets: x, .. }, Cmd::Draw { targets: y, .. }) => x.same(y),
            _ => false,
        };
        // A draw or dispatch that reads a root parameter the title never set is dropped by itself:
        // refusing the whole piece for it also dropped the piece's value writes, and a queue that
        // waits for one of those then holds everything behind it for good. Any other error still
        // refuses the piece. A dropped command still ends the pass it would have ended, so the
        // commands either side that it kept apart stay in two passes (`cut`).
        let mut cmds: Vec<&Cmd<B>> = Vec::new();
        let mut preps = Vec::new();
        let mut cut: Vec<bool> = Vec::new();
        let mut dropped: Vec<&Cmd<B>> = Vec::new();
        for c in lists.iter().flat_map(|l| l.cmds.iter()) {
            let no_target = match c {
                Cmd::Draw { targets, .. } => {
                    targets.color.is_empty() && targets.depth.is_none()
                }
                Cmd::Clear { color, depth } => color.is_none() && depth.is_none(),
                _ => false,
            };
            if no_target {
                self.no_targets.set(self.no_targets.get() + 1);
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!("native host drops work: a draw or clear with no render target the host represents (first time)");
                });
                dropped.push(c);
                continue;
            }
            match self.prep(c) {
                Ok(p) => {
                    cut.push(dropped.drain(..).any(|d| !same(d, c)));
                    cmds.push(c);
                    preps.push(p);
                }
                Err(e)
                    if e.ends_with(NEVER_SET)
                        && matches!(c, Cmd::Draw { .. } | Cmd::Dispatch { .. }) =>
                {
                    self.unset_roots.set(self.unset_roots.get() + 1);
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| {
                        tracing::warn!(error = %e, "native host drops work: a draw or dispatch that reads a root parameter the title never set (first time)");
                    });
                    dropped.push(c);
                }
                Err(e) => return Err(e),
            }
        }

        let mut groups = vec![];
        let mut i = 0;
        while i < cmds.len() {
            let mut j = i + 1;
            while j < cmds.len() && !cut[j] && same(cmds[i], cmds[j]) {
                j += 1;
            }
            groups.push(match cmds[i] {
                Cmd::Dispatch { .. } => Group::Compute(i, j),
                Cmd::Copy { .. } | Cmd::Fill { .. } => Group::Blit(i, j),
                Cmd::CopyTexture { .. } | Cmd::CopyTextureRegion { .. } => Group::Blit(i, j),
                Cmd::Draw { .. } => Group::Render(i, j),
                Cmd::Clear { .. } => {
                    j = i + 1;
                    Group::Clear(i)
                }
            });
            i = j;
        }

        // Every pool buffer the work reaches is resident for the queue's work from now on (once).
        let mut buffers: Vec<&B::Buffer> = Vec::new();
        for p in &preps {
            buffers.extend(p.bufs.iter().map(|b| &**b));
            buffers.extend(p.vertex.iter().map(|b| &*b.0));
            buffers.extend(p.index.iter().map(|b| &*b.0));
            buffers.extend(p.blit.iter().map(|b| &*b.0));
        }
        if let Err(e) = self.backend.make_resident(queue, &buffers) {
            tracing::warn!(error = %e, "pool buffers were not made resident");
        }

        let compute: Vec<Vec<Dispatch<'_, B>>> = groups
            .iter()
            .map(|g| match g {
                Group::Compute(a, b) => (*a..*b)
                    .map(|k| {
                        let Cmd::Dispatch {
                            pso,
                            heaps: h,
                            groups,
                            ..
                        } = cmds[k]
                        else {
                            unreachable!()
                        };
                        Dispatch {
                            pipeline: &pso.pso,
                            root: preps[k].root.clone(),
                            resources: preps[k].bufs.iter().map(|b| &**b).collect(),
                            heaps: match &preps[k].heap {
                                Some(m) => Heaps {
                                    resource: Some(&m.heap),
                                    sampler: h.1.as_ref().map(|s| &s.heap),
                                },
                                None => heaps(h),
                            },
                            groups: *groups,
                            threads_per_group: pso.threads,
                        }
                    })
                    .collect(),
                _ => vec![],
            })
            .collect();

        let mut passes: Vec<Pass<'_, B>> = vec![];
        for (n, g) in groups.iter().enumerate() {
            passes.push(match g {
                Group::Compute(..) => Pass::Compute(&compute[n]),
                Group::Blit(a, b) => Pass::Blit(
                    (*a..*b)
                        .map(|k| match cmds[k] {
                            Cmd::Copy { len, .. } => {
                                let p = &preps[k].blit;
                                BlitOp::Copy {
                                    src: (&*p[0].0, p[0].1),
                                    dst: (&*p[1].0, p[1].1),
                                    len: *len,
                                }
                            }
                            Cmd::Fill { len, value, .. } => {
                                let p = &preps[k].blit[0];
                                BlitOp::Fill {
                                    dst: (&*p.0, p.1),
                                    len: *len,
                                    value: *value,
                                }
                            }
                            Cmd::CopyTexture { .. } => match preps[k].texture.as_ref() {
                                Some(TextureOp::Copy { src, dst }) => BlitOp::CopyTexture {
                                    src: &**src,
                                    dst: &**dst,
                                },
                                _ => unreachable!(),
                            },
                            Cmd::CopyTextureRegion { .. } => match preps[k].texture.as_ref() {
                                Some(TextureOp::Region {
                                    src,
                                    dst,
                                    src_slice,
                                    src_level,
                                    src_origin,
                                    size,
                                    dst_slice,
                                    dst_level,
                                    dst_origin,
                                }) => BlitOp::CopyTextureRegion {
                                    src: &**src,
                                    dst: &**dst,
                                    src_slice: *src_slice,
                                    src_level: *src_level,
                                    src_origin: *src_origin,
                                    size: *size,
                                    dst_slice: *dst_slice,
                                    dst_level: *dst_level,
                                    dst_origin: *dst_origin,
                                },
                                _ => unreachable!(),
                            },
                            _ => unreachable!(),
                        })
                        .collect(),
                ),
                Group::Clear(k) => {
                    let Cmd::Clear { color, depth } = cmds[*k] else {
                        unreachable!()
                    };
                    Pass::Render(RenderPass {
                        color: color
                            .iter()
                            .map(|(t, c)| ColorTarget {
                                texture: &**t,
                                clear: Some(*c),
                            })
                            .collect(),
                        depth: depth.as_ref().map(|(t, d)| DepthTarget {
                            texture: &**t,
                            clear: Some(*d),
                        }),
                        draws: vec![],
                    })
                }
                Group::Render(a, b) => {
                    let Cmd::Draw {
                        targets: Targets { color, depth },
                        ..
                    } = cmds[*a]
                    else {
                        unreachable!()
                    };
                    Pass::Render(RenderPass {
                        color: color
                            .iter()
                            .map(|t| ColorTarget {
                                texture: &**t,
                                clear: None,
                            })
                            .collect(),
                        depth: depth.as_ref().map(|t| DepthTarget {
                            texture: &**t,
                            clear: None,
                        }),
                        draws: (*a..*b)
                            .map(|k| {
                                let Cmd::Draw {
                                    pso,
                                    topology,
                                    heaps: h,
                                    state,
                                    call,
                                    ..
                                } = cmds[k]
                                else {
                                    unreachable!()
                                };
                                Draw {
                                    pipeline: &**pso,
                                    topology: *topology,
                                    state: *state,
                                    root: preps[k].root.clone(),
                                    fragment_root: preps[k].ps_root.clone(),
                                    heaps: match &preps[k].heap {
                                        Some(m) => Heaps {
                                            resource: Some(&m.heap),
                                            sampler: h.1.as_ref().map(|s| &s.heap),
                                        },
                                        None => heaps(h),
                                    },
                                    resources: preps[k].bufs.iter().map(|b| &**b).collect(),
                                    vertex_buffers: preps[k]
                                        .vertex
                                        .iter()
                                        .map(|(b, p)| (&**b, *p))
                                        .collect(),
                                    index_buffer: preps[k].index.as_ref().map(|(b, p)| (&**b, *p)),
                                    call: *call,
                                }
                            })
                            .collect(),
                    })
                }
            });
        }
        self.passes.set(self.passes.get() + passes.len() as u64);
        self.backend.submit_passes(
            queue,
            &PassSubmission {
                waits,
                passes: &passes,
                signals,
            },
        )
    }
}

/// The title texture a view names (macOS only; the target trace's).
#[cfg(target_os = "macos")]
fn texture_info_of(r: usize) -> Option<super::textures::TextureInfo> {
    crate::resource::texture_info(r as crate::object::This)
}

#[cfg(not(target_os = "macos"))]
fn texture_info_of(_: usize) -> Option<super::textures::TextureInfo> {
    None
}

#[cfg(target_os = "macos")]
fn placed_info_of(address: u64, res: &[u32; 14]) -> Option<super::textures::TextureInfo> {
    crate::resource::placed_info(address, res)
}

#[cfg(not(target_os = "macos"))]
fn placed_info_of(_: u64, _: &[u32; 14]) -> Option<super::textures::TextureInfo> {
    None
}

/// Entries of a bound table at root `slot` to check for plain-store updates at submission:
/// at least `floor`, extended to cover any higher table entry declared or read by the bound
/// shader(s).
fn table_check_count(slot: usize, binding: Option<&Binding>, floor: usize) -> usize {
    let mut count = floor;
    let Ok(slot) = u8::try_from(slot) else {
        return count;
    };
    let mut scan = |b: &Binding| {
        for &(s, n) in &b.table_entries {
            if s == slot {
                count = count.max(n);
            }
        }
    };
    if let Some(b) = binding {
        scan(b);
        if let Some(ps) = &b.ps {
            scan(ps);
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_check_count_uses_shader_metadata_above_default() {
        assert_eq!(table_check_count(0, None, 16), 16);
        let vs = Binding {
            table_entries: vec![(0, 4), (2, 24)],
            ps: Some(Rc::new(Binding {
                table_entries: vec![(2, 30), (3, 19)],
                ..Binding::default()
            })),
            ..Binding::default()
        };
        assert_eq!(table_check_count(0, Some(&vs), 16), 16);
        assert_eq!(table_check_count(1, Some(&vs), 16), 16);
        assert_eq!(table_check_count(2, Some(&vs), 16), 30);
        assert_eq!(table_check_count(3, Some(&vs), 16), 19);
    }
}
