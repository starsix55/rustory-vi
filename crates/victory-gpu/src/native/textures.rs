//! Title textures on the native host: the console's tiled layouts (`victory_formats::texture`)
//! detiled from title memory into backend textures.
//!
//! A title texture lives in title memory in the console's swizzled layout; the backend's texture is
//! Apple's own. Each texture the host samples is detiled into a linear staging buffer and uploaded,
//! mip by mip and slice by slice, when the memory's content changed since the last time (a hash of
//! the memory, taken when work is submitted). Render targets and depth targets are the GPU's: they
//! start empty and are not read from memory.
//!
//! shortcut: the detile runs on the CPU (`Layout::detile`, the code the D3D12 host uses too) and
//! a thread hashes the memory of every sampled texture round after round (see `Watch`); a compute detile that reads the
//! pool buffer directly and a per-page change test (what `watch.rs` gives the D3D12 host) are the
//! next step. A change detiles the whole texture.

use super::descriptors::format;
use super::host::NativeHost;
use super::list::{Binding, Cmd, NativeHeap, NativeList, Root};
use super::title::{Stats, Title};
use std::collections::HashMap;
use std::rc::Rc;
use victory_backend::resources::*;
use victory_formats::texture::{Dimension, Layout};

/// `D3D12_RESOURCE_FLAGS`.
const RENDER_TARGET: u32 = 1;
const DEPTH_STENCIL: u32 = 2;
const UNORDERED_ACCESS: u32 = 4;

/// A title texture as the title made it: its description and where its memory is.
#[derive(Clone)]
pub struct TextureInfo {
    pub dimension: Dimension,
    /// DXGI format number, the console's (before `pc_format`).
    pub format: u32,
    pub width: u32,
    pub height: u32,
    /// Array length, or depth for 3D.
    pub depth_or_layers: u32,
    pub flags: u32,
    /// The layout of plane 0 (depth, for depth and stencil).
    pub layout: Layout,
    pub memory: *const u8,
    pub size: usize,
}

impl TextureInfo {
    pub fn render_target(&self) -> bool {
        self.flags & (RENDER_TARGET | DEPTH_STENCIL) != 0
    }

    /// Identity of the title-memory range a host texture is made from. Two
    /// title resources with this identity and the same layout are aliases of
    /// one backing image and must share it.
    pub fn backing_key(&self) -> (usize, usize, u32, u32, u32, u32) {
        (
            self.memory as usize,
            self.size,
            self.format,
            self.width,
            self.height,
            self.flags,
        )
    }

    /// The backend's description of it; `None` for formats or shapes the host does not know.
    pub fn desc(&self) -> Option<TextureDesc> {
        let depth = self.flags & DEPTH_STENCIL != 0;
        let mut format = format(self.format)?;
        // A typeless 32-bit float is a depth target when the texture is one.
        if depth && format == Format::R32Float {
            format = Format::D32Float;
        }
        if depth && format == Format::R16Unorm {
            format = Format::D16Unorm;
        }
        let (dim, d) = match self.dimension {
            Dimension::Tex3d => (TexDim::D3, self.depth_or_layers as usize),
            _ if self.depth_or_layers > 1 => (TexDim::D2Array, self.depth_or_layers as usize),
            _ => (TexDim::D2, 1),
        };
        Some(TextureDesc {
            dim,
            format,
            width: self.width as usize,
            height: self.height.max(1) as usize,
            depth_or_layers: d,
            mips: self.layout.mips.len().max(1),
            render_target: self.render_target(),
            unordered_access: self.flags & UNORDERED_ACCESS != 0,
        })
    }
}

/// Reads `len` bytes of title memory (zeros where it cannot be read: a page the title never mapped).
pub fn read_memory(at: *const u8, len: usize) -> Vec<u8> {
    const CHUNK: usize = 1 << 20;
    let mut out = vec![0u8; len];
    for (i, chunk) in out.chunks_mut(CHUNK).enumerate() {
        // Only an address, checked by the kernel when read.
        let _ = super::safe_read(at.wrapping_add(i * CHUNK), chunk);
    }
    out
}

/// Title texels as the host format holds them: DXGI `B4G4R4A4_UNORM` (115) has blue in the low
/// nibble and alpha in the top one, Metal's `ABGR4Unorm` alpha low and red top. The host's own
/// writes and reads agree with each other, so only bytes from title memory are converted.
pub fn to_host(format: u32, texels: &mut [u8]) {
    if format == 115 {
        for t in texels.as_chunks_mut::<2>().0 {
            *t = u16::from_le_bytes(*t).rotate_left(4).to_le_bytes();
        }
    }
}

/// Detiles every mip of every slice of `tiled` (the texture's whole memory) into `tex`.
pub fn upload<B: ResourceBackend>(
    backend: &B,
    tex: &B::Texture,
    info: &TextureInfo,
    tiled: &[u8],
) -> Result<(), String> {
    let layout = &info.layout;
    let slices = if info.dimension == Dimension::Tex3d {
        1
    } else {
        layout.slices
    };
    let n = layout.desc.element_bytes as usize;
    for mip in 0..layout.mips.len() as u32 {
        let m = &layout.mips[mip as usize];
        let row = m.width as usize * n;
        let slice_pitch = row * m.height as usize;
        let depth = layout.subresource_depth(mip).max(1) as usize;
        let mut linear = vec![0u8; slice_pitch * depth];
        for slice in 0..slices {
            layout
                .detile(tiled, mip, slice, &mut linear, row, slice_pitch)
                .map_err(|e| format!("detile mip {mip} slice {slice}: {e:?}"))?;
            to_host(info.format, &mut linear);
            backend.upload_texture(tex, mip as usize, slice as usize, &linear, row, slice_pitch);
        }
    }
    Ok(())
}

/// Textures whose memory the scanner thread watches, and those it found changed.
///
/// Hashing every sampled texture's memory on every submission (150 ms at 170 MiB, under the
/// title lock the title's other threads wait on) made a title that streams textures slow to a
/// crawl as they accumulated. A thread of its own hashes them in rounds, off the lock, and a
/// submission only looks at the textures it marked.
/// shortcut: a change is seen one round later (a round is as long as the hashing takes, about
/// 100 ms for 150 MiB), so a texture written just before it is sampled may show its old content
/// for a few frames; a per-page write test would make that exact.
#[derive(Default)]
struct Watch {
    /// Title texture key to (address, bytes, whether it is a render target: its first sight is
    /// its baseline, not a change).
    entries: HashMap<usize, (usize, usize, bool)>,
    dirty: std::collections::HashSet<usize>,
    /// Keys a heap, view or draw actually bound. An entry no draw has ever
    /// used is not scanned: no shader can sample it.
    used: std::collections::HashSet<usize>,
}

static WATCH: std::sync::Mutex<Option<Watch>> = std::sync::Mutex::new(None);

fn watch<R>(f: impl FnOnce(&mut Watch) -> R) -> R {
    f(WATCH.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_default())
}

fn start_scanner() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let _ = std::thread::Builder::new().name("victory texture scan".into()).spawn(|| {
            let mut seen: HashMap<usize, blake3::Hash> = HashMap::new();
            let mut buf = vec![0u8; 1 << 20];
            loop {
                let list: Vec<(usize, (usize, usize, bool))> = watch(|w| {
                    w.entries
                        .iter()
                        .filter(|(k, _)| w.used.contains(k))
                        .map(|(&k, &v)| (k, v))
                        .collect()
                });
                seen.retain(|k, _| list.iter().any(|(l, _)| l == k));
                for &(key, (at, size, target)) in &list {
                    let mut h = blake3::Hasher::new();
                    for off in (0..size).step_by(buf.len()) {
                        let n = buf.len().min(size - off);
                        buf[..n].fill(0);
                        let _ = super::safe_read((at + off) as *const u8, &mut buf[..n]);
                        h.update(&buf[..n]);
                    }
                    let h = h.finalize();
                    let before = seen.insert(key, h);
                    if before != Some(h) && !(target && before.is_none()) {
                        watch(|w| w.dirty.insert(key));
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        });
    });
}

/// A texture the host keeps for a title texture.
pub struct Tracked<B: ResourceBackend> {
    pub texture: Rc<B::Texture>,
    info: TextureInfo,
    /// The hash of the memory last uploaded; `None` for targets and before the first upload.
    last: Option<blake3::Hash>,
    /// Uploads of its memory so far.
    pub uploads: u64,
    /// Submissions left in which its memory is probed at submission rather than left to the
    /// scanner (0: cold), and the hash of what the last probe read.
    hot: u32,
    probe: Option<blake3::Hash>,
    /// Whether the GPU has written it (drawn or cleared as a target, a copy destination, or a
    /// UAV). Its contents then do not exist in title memory and it must not be given up.
    gpu_written: bool,
    /// Draws and clears that bound it as a target.
    target_binds: u64,
    /// Fingerprint of `info.layout`: equal layouts can share a backing image
    /// even when the title described the resource with another format.
    layout_id: u64,
}

/// Submissions a texture whose memory changed (or that was uploaded blank) stays hot without a
/// further change.
/// Stable fingerprint of a console layout: equal layouts can share one backing
/// image even when the title described the resource with another format.
fn layout_id(layout: &Layout) -> u64 {
    let hash = blake3::hash(format!("{layout:?}").as_bytes());
    u64::from_le_bytes(hash.as_bytes()[..8].try_into().expect("a blake3 hash has 32 bytes"))
}

const HOT: u32 = 600;
/// Maximum hot textures probed synchronously on a single submission. Boot movies and UI glyph
/// caches have ~12 hot textures (all checked every submission); when world streaming realizes
/// thousands of blank textures at once, this bounds per-submission probing to O(1) while the
/// background scanner thread catches the rest off the host lock.
///
/// shortcut: a title write to a hot texture can take up to `ceil(hot / MAX_SYNC_HOT)`
/// submissions to be noticed synchronously (unless the scanner catches it first); raise this if
/// more than 32 textures need same-submission updates (e.g. many simultaneous streaming UI atlases).
const MAX_SYNC_HOT: usize = 32;
/// Pages of a large hot texture's memory a probe reads, spread over it.
const PROBE_PAGES: usize = 8;
/// Bytes sampled per page when probing a large hot texture.
const PROBE_SAMPLE: usize = 256;
/// Hot textures up to this size that changed more than once are probed whole (a glyph cache gets
/// a few glyphs at a time). Textures written once after a blank upload (streaming) are sampled:
/// probing all of them whole looked about 15% slower in one run.
const PROBE_WHOLE: usize = 4 << 20;

/// Hash of the bytes a probe of `size` bytes at `at` reads: all of them if `whole`, else
/// `PROBE_PAGES` samples of `PROBE_SAMPLE` bytes spread across the texture, reusing `scratch`.
///
/// shortcut: when sampled, a write that misses every sampled window goes unseen until the
/// scanner's next round; a texture the title rewrites whole (a movie's planes) is caught at once.
fn probe_hash(at: usize, size: usize, whole: bool, scratch: &mut Vec<u8>) -> blake3::Hash {
    let mut hasher = blake3::Hasher::new();
    if whole || size <= PROBE_PAGES * PROBE_SAMPLE {
        const CHUNK: usize = 1 << 20;
        let cap = size.min(CHUNK);
        if scratch.len() < cap {
            scratch.resize(cap, 0);
        }
        for off in (0..size).step_by(CHUNK) {
            let n = CHUNK.min(size - off);
            scratch[..n].fill(0);
            let _ = super::safe_read((at + off) as *const u8, &mut scratch[..n]);
            hasher.update(&scratch[..n]);
        }
        return hasher.finalize();
    }
    const PAGE: usize = 4096;
    let pages = size.div_ceil(PAGE);
    if scratch.len() < PROBE_SAMPLE {
        scratch.resize(PROBE_SAMPLE, 0);
    }
    for k in 0..PROBE_PAGES {
        let off = k * pages / PROBE_PAGES * PAGE;
        let n = PROBE_SAMPLE.min(size - off);
        scratch[..n].fill(0);
        let _ = super::safe_read((at + off) as *const u8, &mut scratch[..n]);
        hasher.update(&scratch[..n]);
    }
    hasher.finalize()
}

/// The key of a texture a view names by address and description (no resource object): a number no
/// object address has (the top bit is set).
pub fn placed_key(address: u64, description: &[u32; 14]) -> usize {
    let mut h = blake3::Hasher::new();
    h.update(&address.to_le_bytes());
    for (i, w) in description.iter().enumerate() {
        // Words 1 and 13 are padding (stack bytes in the title's records).
        h.update(&(if i == 1 || i == 13 { 0 } else { *w }).to_le_bytes());
    }
    let b = h.finalize();
    (u64::from_le_bytes(b.as_bytes()[..8].try_into().expect("8 bytes")) | 1 << 63) as usize
}

impl<B: PresentBackend> Title<B> {
    /// A title texture exists. Its host texture is made when a view, a render target or a clear
    /// first names it (most of a title's textures are never used), and filled from title memory
    /// when work next goes to the GPU.
    pub fn texture_created(&mut self, key: usize, info: TextureInfo) {
        self.pending.insert(key, info);
    }

    /// Makes the host texture of title texture `key` if it has none yet and registers it for
    /// views. `false` when there is none (not a title texture, or one the host cannot represent).
    pub fn realize(&mut self, key: usize) -> bool {
        let Self {
            host,
            pending,
            textures,
            backing,
            unloaded,
            ..
        } = self;
        realize_key(host, pending, textures, backing, unloaded, key)
    }

    /// The format of the host texture of title texture `key`.
    pub fn format_of(&self, key: usize) -> Option<Format> {
        self.textures.get(&key).and_then(|t| t.info.desc()).map(|d| d.format)
    }

    /// The host texture of title texture `key`.
    pub fn texture_of(&mut self, key: usize) -> Option<Rc<B::Texture>> {
        self.realize(key);
        self.mark_used(key);
        self.textures.get(&key).map(|t| t.texture.clone())
    }

    /// Marks a texture as bound by a heap, view or draw: from now on the
    /// scanner hashes it and it is uploaded when its memory changes.
    pub fn mark_used(&mut self, key: usize) {
        if self.textures.contains_key(&key) {
            watch(|w| w.used.insert(key));
        }
    }

    /// The GPU wrote `key` (a target bind, a copy destination, a UAV): its
    /// contents no longer come from title memory and it cannot be given up.
    pub fn mark_gpu_written(&mut self, key: usize) {
        if let Some(t) = self.textures.get_mut(&key) {
            t.gpu_written = true;
        }
    }

    /// A draw or clear bound `key` as a render target: it has been written,
    /// and is counted for the usage report.
    pub fn mark_target(&mut self, key: usize) {
        if let Some(t) = self.textures.get_mut(&key) {
            t.gpu_written = true;
            t.target_binds += 1;
        }
    }

    /// Host textures made, the bytes of title memory they duplicate, and how
    /// many of those are linear textures that could be views of title memory.
    pub fn resident(&self) -> (usize, usize, usize, usize) {
        let mut linear = (0, 0);
        let mut seen = std::collections::HashSet::new();
        let mut bytes = 0;
        for t in self.textures.values() {
            let ptr = Rc::as_ptr(&t.texture) as usize;
            if !seen.insert(ptr) {
                continue;
            }
            bytes += t.info.size;
            if t.info.layout.desc.swizzle == victory_formats::texture::Swizzle::LINEAR {
                linear.0 += 1;
                linear.1 += t.info.size;
            }
        }
        (self.textures.len(), bytes, linear.0, linear.1)
    }

    /// Bytes the backend actually allocated for those textures, counting one
    /// backing image once however many title keys alias it.
    pub fn allocated(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        let mut bytes = 0;
        for t in self.textures.values() {
            if seen.insert(Rc::as_ptr(&t.texture) as usize) {
                bytes += self.host.backend.allocated_size(&t.texture);
            }
        }
        bytes
    }

    /// Title bytes summed over every realized key, aliases included: what
    /// one host texture per key would have duplicated.
    pub fn key_bytes(&self) -> usize {
        self.textures.values().map(|t| t.info.size).sum()
    }

    /// Ways the realized set still duplicates backing storage:
    /// `(same-layout groups with different resource descriptions, extra keys,
    /// extra bytes, bytes in overlapping non-identical ranges)`. Equal
    /// layouts with different formats could share one image through views;
    /// overlapping ranges need a range index before they can share.
    pub fn alias_opportunities(&self) -> (usize, usize, usize, usize) {
        let mut groups: HashMap<(usize, usize, u64), Vec<&Tracked<B>>> = HashMap::new();
        let mut unique: Vec<(usize, usize, u64)> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for t in self.textures.values() {
            let at = t.info.memory as usize;
            let key = (at, t.info.size, t.layout_id);
            groups.entry(key).or_default().push(t);
            if seen.insert(Rc::as_ptr(&t.texture) as usize) {
                unique.push(key);
            }
        }
        let (mut multi, mut extra_keys, mut extra_bytes) = (0, 0, 0);
        for (key, members) in &groups {
            let mut descs = std::collections::HashSet::new();
            for t in members {
                descs.insert((t.info.format, t.info.width, t.info.height, t.info.flags));
            }
            if descs.len() > 1 {
                multi += 1;
                extra_keys += descs.len() - 1;
                extra_bytes += key.1 * (descs.len() - 1);
            }
        }
        unique.sort_unstable();
        let (mut overlaps, mut overlap_bytes, mut max_end) = (0, 0, 0);
        for &(at, size, _) in &unique {
            if at < max_end {
                overlaps += 1;
                overlap_bytes += size.min(max_end - at);
            }
            max_end = max_end.max(at + size);
        }
        (multi, extra_keys, extra_bytes, overlap_bytes)
    }

    /// Title bytes of the given keys, counting each backing image once.
    pub fn stale_bytes(&self, keys: &[usize]) -> usize {
        let mut seen = std::collections::HashSet::new();
        let mut bytes = 0;
        for key in keys {
            if let Some(t) = self.textures.get(key)
                && seen.insert(Rc::as_ptr(&t.texture) as usize)
            {
                bytes += t.info.size;
            }
        }
        bytes
    }

    /// Keys of host textures per backing image.
    fn holders(&self) -> HashMap<usize, usize> {
        let mut holders: HashMap<usize, usize> = HashMap::new();
        for t in self.textures.values() {
            *holders.entry(Rc::as_ptr(&t.texture) as usize).or_default() += 1;
        }
        holders
    }

    /// Whether `t`, whose backing is held by `holders` keys, has a strong
    /// reference from anywhere but this map and the title-resource map (two
    /// per key: a live heap, view, draw or copy keeps it).
    fn internally_held(t: &Tracked<B>, holders: usize) -> bool {
        Rc::strong_count(&t.texture) <= holders * 2
    }

    /// Host textures whose only strong references are this map, the
    /// title-resource map and no live descriptor heap entry (no live view or
    /// heap keeps them).
    pub fn unreferenced(&self) -> usize {
        let holders = self.holders();
        let pinned = self.heap_pinned();
        self.textures
            .iter()
            .filter(|(_, t)| {
                let ptr = Rc::as_ptr(&t.texture) as usize;
                !pinned.contains(&ptr)
                    && Self::internally_held(t, holders.get(&ptr).copied().unwrap_or(1))
            })
            .count()
    }

    /// Backing images a live descriptor heap entry names. The entry retains
    /// the Metal texture, so the device memory stays allocated even if this
    /// map drops the key.
    fn heap_pinned(&self) -> std::collections::HashSet<usize> {
        let refs = self.host.heap_texture_refs();
        self.textures
            .iter()
            .filter(|(k, _)| refs.contains_key(k))
            .map(|(_, t)| Rc::as_ptr(&t.texture) as usize)
            .collect()
    }

    /// Unique host backing bytes by kind (depth, render target, unordered
    /// access, sampled) as (kind, keys, bytes), and the largest backing
    /// images, for deciding what can actually be given up.
    pub fn breakdown(&self) -> (Vec<(&'static str, usize, usize)>, Vec<String>) {
        let holders = self.holders();
        let refs = self.host.heap_texture_refs();
        let pinned = self.heap_pinned();
        let mut seen = std::collections::HashSet::new();
        let mut kinds: Vec<(&'static str, usize, usize)> = vec![
            ("depth", 0, 0),
            ("render-target", 0, 0),
            ("unordered", 0, 0),
            ("sampled", 0, 0),
            ("gpu-written", 0, 0),
            ("target-never-written", 0, 0),
            ("heap-referenced", 0, 0),
        ];
        let mut top: Vec<(usize, String)> = Vec::new();
        for (&key, t) in &self.textures {
            let ptr = Rc::as_ptr(&t.texture) as usize;
            let ext = Rc::strong_count(&t.texture)
                .saturating_sub(holders.get(&ptr).copied().unwrap_or(1) * 2);
            let i = if t.info.flags & DEPTH_STENCIL != 0 {
                0
            } else if t.info.flags & RENDER_TARGET != 0 {
                1
            } else if t.info.flags & UNORDERED_ACCESS != 0 {
                2
            } else {
                3
            };
            kinds[i].1 += 1;
            if refs.contains_key(&key) {
                kinds[6].1 += 1;
            }
            if t.gpu_written {
                kinds[4].1 += 1;
            } else if t.info.render_target() {
                kinds[5].1 += 1;
            }
            if seen.insert(ptr) {
                kinds[i].2 += t.info.size;
                if pinned.contains(&ptr) {
                    kinds[6].2 += t.info.size;
                }
                if t.gpu_written {
                    kinds[4].2 += t.info.size;
                } else if t.info.render_target() {
                    kinds[5].2 += t.info.size;
                }
                top.push((
                    t.info.size,
                    format!(
                        "{:#x}+{} {}x{}x{} fmt={} flags={:#x} mips={} ext={} uploads={} binds={} gpu={} heap_refs={} key={key:#x}",
                        t.info.memory as usize,
                        t.info.size,
                        t.info.width,
                        t.info.height,
                        t.info.depth_or_layers,
                        t.info.format,
                        t.info.flags,
                        t.info.layout.mips.len(),
                        ext,
                        t.uploads,
                        t.target_binds,
                        t.gpu_written,
                        refs.get(&key).copied().unwrap_or(0),
                    ),
                ));
            }
        }
        top.sort_by_key(|(size, _)| std::cmp::Reverse(*size));
        top.truncate(12);
        (kinds, top.into_iter().map(|(_, s)| s).collect())
    }

    /// How many times title texture `key`'s memory was detiled into its host texture.
    pub fn uploads_of(&self, key: usize) -> u64 {
        self.textures.get(&key).map_or(0, |t| t.uploads)
    }

    /// Gives up reconstructible host textures until their unique backing total
    /// is within `limit` bytes. A descriptor entry naming one is repointed at
    /// the spare first; `realize_pending` restores it before the next list that
    /// can index its heap is encoded. Heaps used recently or still in flight
    /// are left alone, so active bindless sets are not thrashed.
    pub fn evict_textures(&mut self, limit: usize) -> (usize, usize) {
        /// Frames a heap stays hot after a list last named it.
        const KEEP_FRAMES: u64 = 120;
        let frame = crate::frame::current();
        let mut blocked = std::collections::HashSet::new();
        for heap in self.host.heaps_snapshot() {
            let hot = frame.saturating_sub(heap.last_used.get()) < KEEP_FRAMES;
            if hot || self.heap_in_flight(&heap) {
                blocked.extend(heap.keys.borrow().values().copied());
            }
        }
        let blocked_ptrs: std::collections::HashSet<usize> = self
            .textures
            .iter()
            .filter(|(k, _)| blocked.contains(k))
            .map(|(_, t)| Rc::as_ptr(&t.texture) as usize)
            .collect();
        let mut holders = self.holders();
        let mut seen = std::collections::HashSet::new();
        let mut total: usize = 0;
        for t in self.textures.values() {
            // A UAV's contents may have been written through a descriptor this
            // host cannot attribute yet, so it stays pinned. Hot textures (a
            // movie plane, a glyph cache) keep their synchronous probe state.
            if t.gpu_written || t.info.flags & UNORDERED_ACCESS != 0 || t.hot > 0 {
                continue;
            }
            let ptr = Rc::as_ptr(&t.texture) as usize;
            if blocked_ptrs.contains(&ptr) {
                continue;
            }
            if seen.insert(ptr) {
                total += t.info.size;
            }
        }
        if total <= limit {
            return (0, 0);
        }
        let candidates: Vec<usize> = self
            .textures
            .iter()
            .filter(|(_, t)| {
                if t.gpu_written || t.info.flags & UNORDERED_ACCESS != 0 || t.hot > 0 {
                    return false;
                }
                let ptr = Rc::as_ptr(&t.texture) as usize;
                !blocked_ptrs.contains(&ptr)
                    && Self::internally_held(t, holders.get(&ptr).copied().unwrap_or(1))
            })
            .map(|(&k, _)| k)
            .collect();
        let (mut n, mut bytes) = (0, 0);
        for key in candidates {
            if total <= limit {
                break;
            }
            let Some(format) = self
                .textures
                .get(&key)
                .map(|t| t.info.desc().map_or(Format::Rgba8Unorm, |d| d.format))
            else {
                continue;
            };
            if !self.host.evict_texture_key(key, format) {
                continue;
            }
            let Some(t) = self.textures.remove(&key) else {
                continue;
            };
            let ptr = Rc::as_ptr(&t.texture) as usize;
            let size = t.info.size;
            self.pending.insert(key, t.info);
            self.unloaded.remove(&key);
            self.hot_keys.retain(|&k| k != key);
            watch(|w| {
                w.entries.remove(&key);
                w.dirty.remove(&key);
                w.used.remove(&key);
            });
            // The backing image is only freed when its last key goes.
            let left = holders.entry(ptr).or_default();
            *left = left.saturating_sub(1);
            if *left == 0 {
                total = total.saturating_sub(size);
                bytes += size;
            }
            n += 1;
        }
        (n, bytes)
    }

    pub fn texture_released(&mut self, key: usize) {
        self.host.release_texture_key(key);
        self.textures.remove(&key);
        self.pending.remove(&key);
        self.unloaded.remove(&key);
        self.hot_keys.retain(|&k| k != key);
        watch(|w| {
            w.entries.remove(&key);
            w.dirty.remove(&key);
            w.used.remove(&key);
        });
    }

    /// Brings the textures the host samples up to date with title memory: those whose memory
    /// changed since their last upload are detiled again.
    pub fn sync_textures(&mut self) {
        let keys: Vec<usize> = self
            .textures
            .iter()
            .filter(|(_, t)| !t.info.render_target())
            .map(|(&k, _)| k)
            .collect();
        self.sync_keys(keys);
    }

    /// What a submission does: the textures not uploaded yet, and those the scanner or hot probe
    /// found changed since.
    pub fn sync_dirty(&mut self) {
        // Host textures duplicate title memory. Sampled textures with no
        // live heap/view reference are safe to give up; their title memory
        // stays the source of truth (VICTORY_TEXTURE_BUDGET_MB, default 2048).
        static BUDGET: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let limit = *BUDGET.get_or_init(|| {
            std::env::var("VICTORY_TEXTURE_BUDGET_MB")
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(2048)
                .saturating_mul(1024 * 1024)
        });
        let (evicted, freed) = {
            let frame = crate::frame::current();
            if self.last_evict == frame {
                (0, 0)
            } else {
                self.last_evict = frame;
                self.evict_textures(limit)
            }
        };
        if evicted > 0 && std::env::var_os("VICTORY_TRACE_TEXTURE_MEMORY").is_some() {
            tracing::info!(evicted, freed, limit, "host texture budget eviction");
        }
        let mut dirty = watch(|w| std::mem::take(&mut w.dirty));
        // Hot textures (a movie's planes, a glyph cache) are probed synchronously in O(1) bounded
        // batches so world streaming (thousands of blank textures) never stalls submissions.
        let mut budget = self.hot_keys.len().min(MAX_SYNC_HOT);
        while budget > 0 && !self.hot_keys.is_empty() {
            budget -= 1;
            let idx = self.hot_cursor % self.hot_keys.len();
            let key = self.hot_keys[idx];
            let Some(t) = self.textures.get_mut(&key) else {
                self.hot_keys.swap_remove(idx);
                continue;
            };
            if t.hot == 0 {
                self.hot_keys.swap_remove(idx);
                continue;
            }
            let whole = t.uploads >= 2 && t.info.size <= PROBE_WHOLE;
            let now = probe_hash(t.info.memory as usize, t.info.size, whole, &mut self.probe_buf);
            let before = t.probe.replace(now);
            if before != Some(now) {
                dirty.insert(key);
                self.hot_cursor = idx + 1;
            } else {
                t.hot -= 1;
                if t.hot == 0 {
                    t.probe = None;
                    self.hot_keys.swap_remove(idx);
                } else {
                    self.hot_cursor = idx + 1;
                }
            }
        }
        let unloaded = std::mem::take(&mut self.unloaded);
        if unloaded.is_empty() && dirty.is_empty() {
            return;
        }
        let mut todo: Vec<usize> = unloaded.into_iter().collect();
        for key in dirty {
            if !todo.contains(&key) {
                todo.push(key);
            }
        }
        self.sync_keys(todo);
    }

    fn sync_keys(&mut self, keys: Vec<usize>) {
        let Title {
            host,
            textures,
            hot_keys,
            stats,
            ..
        } = self;
        for key in keys {
            let Some(t) = textures.get_mut(&key) else {
                continue;
            };
            let tiled = read_memory(t.info.memory, t.info.size);
            let hash = blake3::hash(&tiled);
            if t.last == Some(hash) {
                continue;
            }
            // Written again after its first upload, or blank at it (written later): hot.
            if !t.info.render_target() && (t.uploads > 0 || tiled.iter().all(|&b| b == 0)) {
                if t.hot == 0 {
                    hot_keys.push(key);
                }
                t.hot = HOT;
            }
            match upload(&host.backend, &t.texture, &t.info, &tiled) {
                Ok(()) => {
                    t.last = Some(hash);
                    t.uploads += 1;
                    stats.detiles += 1;
                }
                Err(e) => {
                    t.last = Some(hash);
                    stats.undetilable += 1;
                    tracing::warn!(texture = key, error = %e, "a title texture was not detiled");
                }
            }
        }
    }
}

/// The textures of a title host, by title resource key.
pub type Textures<B: ResourceBackend> = HashMap<usize, Tracked<B>>;

/// Host textures by canonical title-memory range and layout.
pub type BackingMap<B> = HashMap<
    (usize, usize, u32, u32, u32, u32),
    (TextureInfo, std::rc::Weak<<B as ResourceBackend>::Texture>),
>;

/// Uploads one newly realized texture from title memory immediately. It was
/// realized after `sync_dirty` for the submission, so waiting for the next
/// scanner round would sample an uninitialized host texture.
fn upload_key<B: PresentBackend>(
    host: &NativeHost<B>,
    textures: &mut Textures<B>,
    stats: &mut Stats,
    key: usize,
) {
    let Some(t) = textures.get_mut(&key) else {
        return;
    };
    let tiled = read_memory(t.info.memory, t.info.size);
    let hash = blake3::hash(&tiled);
    if t.last == Some(hash) {
        return;
    }
    match upload(&host.backend, &t.texture, &t.info, &tiled) {
        Ok(()) => {
            t.last = Some(hash);
            t.uploads += 1;
            stats.detiles += 1;
        }
        Err(e) => {
            t.last = Some(hash);
            stats.undetilable += 1;
            tracing::warn!(texture = key, error = %e, "a pending title texture was not detiled");
        }
    }
}

/// One heap's reachable descriptor entries in this list: the whole heap for
/// computed indices, or the union of fixed ranges the shaders can read.
pub(super) struct HeapAccess<B: PresentBackend> {
    pub(super) heap: Rc<NativeHeap<B>>,
    pub(super) whole: bool,
    pub(super) ranges: Vec<(usize, usize)>,
}

impl<B: PresentBackend> HeapAccess<B> {
    fn covers(&self, index: usize) -> bool {
        self.whole || self.ranges.iter().any(|&(a, b)| index >= a && index < b)
    }
}

pub(super) struct Accesses<B: PresentBackend> {
    pub(super) list: Vec<HeapAccess<B>>,
    by_ptr: HashMap<usize, usize>,
}

impl<B: PresentBackend> Default for Accesses<B> {
    fn default() -> Self {
        Accesses {
            list: Vec::new(),
            by_ptr: HashMap::new(),
        }
    }
}

impl<B: PresentBackend> Accesses<B> {
    fn get(&mut self, heap: &Rc<NativeHeap<B>>) -> &mut HeapAccess<B> {
        let ptr = Rc::as_ptr(heap) as usize;
        if let Some(&i) = self.by_ptr.get(&ptr) {
            return &mut self.list[i];
        }
        let i = self.list.len();
        self.list.push(HeapAccess {
            heap: heap.clone(),
            whole: false,
            ranges: Vec::new(),
        });
        self.by_ptr.insert(ptr, i);
        &mut self.list[i]
    }

    fn whole(&mut self, heap: &Rc<NativeHeap<B>>) {
        self.get(heap).whole = true;
    }

    fn range(&mut self, heap: &Rc<NativeHeap<B>>, start: usize, end: usize) {
        if end > start {
            self.get(heap).ranges.push((start, end));
        }
    }
}

/// Adds the fixed table ranges a binding reads. Dynamic slots mark the whole
/// heap reachable; a slot absent from both lists is not read by the shader.
pub(super) fn note_binding<B: PresentBackend>(
    accesses: &mut Accesses<B>,
    root: &[Option<Root<B>>],
    binding: &Binding,
) {
    for &(slot, count) in &binding.table_entries {
        let Some(Root::Table(heap, at)) = root.get(usize::from(slot)).and_then(|r| r.as_ref()) else {
            continue;
        };
        let end = at.saturating_add(count).min(heap.count);
        if binding.table_dynamic.contains(&slot) {
            accesses.whole(heap);
        } else {
            accesses.range(heap, *at, end);
        }
    }
    for &slot in &binding.table_dynamic {
        if let Some(Root::Table(heap, _)) = root.get(usize::from(slot)).and_then(|r| r.as_ref()) {
            accesses.whole(heap);
        }
    }
    if let Some(ps) = &binding.ps {
        note_binding(accesses, root, ps);
    }
}

fn whole_root_tables<B: PresentBackend>(
    accesses: &mut Accesses<B>,
    root: &[Option<Root<B>>],
) {
    for r in root.iter().flatten() {
        if let Root::Table(heap, _) = r {
            accesses.whole(heap);
        }
    }
}

/// Before a list is encoded: refresh descriptor tables, then realize the
/// pending entries the list's shaders can actually reach and remirror their
/// slots. Fixed table accesses narrow the set to their ranges; computed
/// indices force the whole heap. This keeps dynamic indexing correct without
/// realizing every texture in a large bound heap.
#[allow(clippy::too_many_arguments)]
pub(super) fn realize_pending<B: PresentBackend>(
    host: &NativeHost<B>,
    pending: &mut HashMap<usize, TextureInfo>,
    textures: &mut Textures<B>,
    backing: &mut BackingMap<B>,
    unloaded: &mut std::collections::HashSet<usize>,
    stats: &mut Stats,
    queue: usize,
    submitted: u64,
    lists: &[Rc<NativeList<B>>],
) -> Result<Vec<Rc<NativeHeap<B>>>, String> {
    let conservative = std::env::var_os("VICTORY_CONSERVATIVE_TEXTURES").is_some();
    let mut accesses = Accesses::default();
    let mut copies: Vec<usize> = Vec::new();
    for list in lists {
        for cmd in &list.cmds {
            match cmd {
                Cmd::Draw {
                    root,
                    binding,
                    heaps: (resource, sampler),
                    ..
                } => {
                    host.remirror_tables(root, binding.as_deref());
                    if conservative || binding.is_none() {
                        if let Some(h) = resource {
                            accesses.whole(h);
                        }
                        if let Some(h) = sampler {
                            accesses.whole(h);
                        }
                        whole_root_tables(&mut accesses, root);
                    } else if let Some(b) = binding.as_deref() {
                        if b.heap || b.ps.as_ref().is_some_and(|ps| ps.heap) {
                            if let Some(h) = resource {
                                accesses.whole(h);
                            }
                            whole_root_tables(&mut accesses, root);
                        }
                        note_binding(&mut accesses, root, b);
                    }
                }
                Cmd::Dispatch {
                    pso, root, heaps, ..
                } => {
                    host.remirror_tables(root, pso.binding.as_ref());
                    if conservative || pso.binding.is_none() {
                        if let Some(h) = &heaps.0 {
                            accesses.whole(h);
                        }
                        if let Some(h) = &heaps.1 {
                            accesses.whole(h);
                        }
                        whole_root_tables(&mut accesses, root);
                    } else if let Some(b) = pso.binding.as_ref() {
                        if b.heap {
                            if let Some(h) = &heaps.0 {
                                accesses.whole(h);
                            }
                            whole_root_tables(&mut accesses, root);
                        }
                        note_binding(&mut accesses, root, b);
                    }
                }
                Cmd::CopyTexture { dst, src } | Cmd::CopyTextureRegion { dst, src, .. } => {
                    copies.push(*dst);
                    copies.push(*src);
                }
                _ => {}
            }
        }
    }

    // A command buffer for this list is about to be committed. Entries in
    // these heaps may not be repointed until it completes.
    let frame = crate::frame::current();
    for access in &accesses.list {
        let heap = &access.heap;
        heap.last_used.set(frame);
        heap.last_queue.set(queue);
        heap.last_serial.set(submitted.saturating_add(1));
        if !heap.marked.get() {
            watch(|w| {
                for &key in heap.keys.borrow().values() {
                    w.used.insert(key);
                }
            });
            heap.marked.set(true);
        }
        if access.whole {
            stats.realize_whole += 1;
            stats.realize_conservative += heap.pending.borrow().len() as u64;
        } else {
            stats.realize_ranged += 1;
            let pending = heap.pending.borrow();
            let covered = pending.iter().filter(|&&i| access.covers(i)).count();
            stats.realize_required += covered as u64;
            stats.realize_deferred += pending.len().saturating_sub(covered) as u64;
            if std::env::var_os("VICTORY_TRACE_REALIZE").is_some() && !pending.is_empty() {
                static N: std::sync::atomic::AtomicU32 =
                    std::sync::atomic::AtomicU32::new(0);
                if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 40 {
                    tracing::info!(
                        heap = format_args!("{:#x}", heap.start),
                        whole = access.whole,
                        ranges = ?access.ranges,
                        pending = pending.len(),
                        covered,
                        entries = heap.keys.borrow().len(),
                        "realize heap"
                    );
                }
            }
        }
    }

    for key in copies {
        if !textures.contains_key(&key) {
            if !realize_key(host, pending, textures, backing, unloaded, key) {
                return Err(format!(
                    "title texture {key:#x} named by a copy could not be realized"
                ));
            }
            upload_key(host, textures, stats, key);
        }
    }

    // Two passes: remirroring a changed title record can introduce a new
    // pending key, which must be realized before the same submission.
    for _ in 0..2 {
        let mut work: Vec<(Rc<NativeHeap<B>>, usize, usize)> = Vec::new();
        for access in &accesses.list {
            let keys = access.heap.keys.borrow();
            for &index in access.heap.pending.borrow().iter() {
                if access.covers(index)
                    && let Some(&key) = keys.get(&index)
                {
                    work.push((access.heap.clone(), index, key));
                }
            }
        }
        if work.is_empty() {
            break;
        }
        let mut progress = false;
        for (heap, index, key) in work {
            if !textures.contains_key(&key) {
                if !realize_key(host, pending, textures, backing, unloaded, key) {
                    return Err(format!(
                        "title texture {key:#x} named by heap {:#x} entry {index} could not be realized",
                        heap.start
                    ));
                }
                upload_key(host, textures, stats, key);
            }
            let handle = heap.start + index as u64 * NativeHost::<B>::step(heap.kind);
            match host.mirror(handle) {
                Ok(true) => progress = true,
                Ok(false) => {
                    return Err(format!(
                        "descriptor {handle:#x} for texture {key:#x} was not remirrored"
                    ));
                }
                Err(e) => {
                    return Err(format!(
                        "descriptor {handle:#x} for texture {key:#x} was not remirrored: {e}"
                    ));
                }
            }
        }
        if !progress {
            break;
        }
    }
    for access in &accesses.list {
        if let Some(&index) = access
            .heap
            .pending
            .borrow()
            .iter()
            .find(|&&i| access.covers(i))
        {
            let key = access.heap.keys.borrow().get(&index).copied().unwrap_or(0);
            return Err(format!(
                "heap {:#x} entry {index} (texture {key:#x}) is still a placeholder",
                access.heap.start
            ));
        }
    }
    Ok(accesses.list.into_iter().map(|a| a.heap).collect())
}

/// Makes the host texture of title texture `key` if it has none yet and
/// registers it for views. Shared by [`Title::realize`] and the encode-time
/// pending realization path, which has the fields split out of `Title`.
pub(super) fn realize_key<B: PresentBackend>(
    host: &NativeHost<B>,
    pending: &mut HashMap<usize, TextureInfo>,
    textures: &mut Textures<B>,
    backing: &mut BackingMap<B>,
    unloaded: &mut std::collections::HashSet<usize>,
    key: usize,
) -> bool {
    if textures.contains_key(&key) {
        return true;
    }
    let Some(info) = pending.remove(&key) else {
        return false;
    };
    let Some(desc) = info.desc() else {
        // Loud once per format: views of it get no texture and their draws sample nothing.
        static SEEN: std::sync::Mutex<Vec<u32>> = std::sync::Mutex::new(Vec::new());
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if !seen.contains(&info.format) {
            seen.push(info.format);
            tracing::warn!(format = info.format, w = info.width, h = info.height, flags = info.flags, "a title texture format the host does not represent (first of this format)");
        }
        return false;
    };
    let bkey = info.backing_key();
    let shared = backing.get(&bkey).and_then(|(old, tex)| {
        (old.layout == info.layout)
            .then(|| tex.upgrade())
            .flatten()
    });
    let tex = match shared {
        Some(t) => t,
        None => {
            let t = match host.texture(&desc) {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!(error = %e, ?desc, "no host texture");
                    return false;
                }
            };
            backing.insert(bkey, (info.clone(), Rc::downgrade(&t)));
            t
        }
    };
    host.register_texture(key, &tex);
    // A descriptor entry that was repointed at the spare while the texture
    // was gone can name the real texture again now.
    host.remirror_texture(key);
    // Render targets too: what the title (CPU, DMA) writes into their memory is uploaded when
    // it changes, as it is not otherwise seen (a GPU write never changes title memory).
    watch(|w| w.entries.insert(key, (info.memory as usize, info.size, info.render_target())));
    start_scanner();
    // Debugging (VICTORY_MAGENTA=1): render targets start magenta, so a frame that stays so
    // was never drawn into, and a black one was drawn black.
    if info.render_target() && std::env::var_os("VICTORY_MAGENTA").is_some() {
        let m = &info.layout.mips[0];
        let n = info.layout.desc.element_bytes as usize;
        let row = m.width as usize * n;
        let pixel = [0xffu8, 0, 0xff, 0xff];
        let linear: Vec<u8> = (0..row * m.height as usize).map(|i| pixel[i % 4]).collect();
        host.backend.upload_texture(&tex, 0, 0, &linear, row, row * m.height as usize);
    }
    if !info.render_target() {
        unloaded.insert(key);
    }
    let layout_id = layout_id(&info.layout);
    textures.insert(
        key,
        Tracked {
            texture: tex,
            info,
            last: None,
            uploads: 0,
            hot: 0,
            probe: None,
            gpu_written: false,
            target_binds: 0,
            layout_id,
        },
    );
    true
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    #[test]
    fn probe_hash_detects_writes_in_sampled_and_whole_modes() {
        let mut mem = vec![0u8; 16 * 4096];
        let mut scratch = Vec::new();
        let h0_sampled = probe_hash(mem.as_ptr() as usize, mem.len(), false, &mut scratch);
        let h0_whole = probe_hash(mem.as_ptr() as usize, mem.len(), true, &mut scratch);
        // Flip a byte in the first sampled window.
        mem[42] = 0xAB;
        let h1_sampled = probe_hash(mem.as_ptr() as usize, mem.len(), false, &mut scratch);
        let h1_whole = probe_hash(mem.as_ptr() as usize, mem.len(), true, &mut scratch);
        assert_ne!(h0_sampled, h1_sampled);
        assert_ne!(h0_whole, h1_whole);
        // Flip a byte outside sampled windows: whole mode catches it.
        mem[42] = 0;
        mem[1000] = 0xCD;
        assert_ne!(h0_whole, probe_hash(mem.as_ptr() as usize, mem.len(), true, &mut scratch));
    }
}
