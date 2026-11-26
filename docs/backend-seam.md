# The backend seam

`victory-backend` is the only contract between the title-facing GPU host
(`victory-gpu`) and a graphics API. `victory-metal` implements it. Nothing
above the seam names Metal, and nothing below it knows about Xbox resources,
title addresses or D3D12.X command streams.

The goal of this document is to keep that boundary explicit so a fix from an
upstream project can be integrated in one place, and so a second backend (a
future Xbox-ABI proxy host, for example) can implement the same contract
without rewriting the title host.

## Layers

| Layer | Crate | Knows about | Must not know about |
|---|---|---|---|
| Title ABI and command recording | `victory-gpu` (`list.rs`, `descriptor.rs`) | D3D12.X calls, title addresses, descriptor records | Metal, device allocation |
| Native host and scheduling | `victory-gpu` (`native/host.rs`, `native/title.rs`) | queues, fences, held work, pipeline cache, texture residency | Metal types |
| Backend contract | `victory-backend` | queues, fences, buffers, textures, heaps, passes | Metal, Xbox, D3D12.X |
| Metal implementation | `victory-metal` | `MTLDevice`, `MTLCommandQueue`, `MTLHeap`, residency | title addresses, D3D12.X |

## Contract slices

### `ComputeBackend`

Queues, fences, zero-copy title-memory buffers, compute pipelines, descriptor
heaps, submissions and completion. `wrap_title_memory` is the zero-copy rule:
the buffer is the guest pages, not a copy.

`memory_usage()` reports device allocation and the recommended working set.
The host logs both; it is the first number to check when a command buffer fails
with `kIOGPUCommandBufferCallbackErrorOutOfMemory`.

`make_resident` is an idempotent convenience. A backend may implement it with
`MTLResidencySet`; it must not become a reason to name buffers on every
encoder, because that would serialize otherwise independent work.

### `ResourceBackend`

Descriptor heaps, textures, samplers and uploads. Additions made for the
hybrid memory work:

- `capabilities()` says which optional allocation paths the backend actually
  implements. Capabilities describe the implementation, not the API: Metal
  supports sparse textures, but until this backend creates one it reports
  `sparse_textures: false`.
- `allocated_size()` is the real device allocation (`MTLAllocation`'s
  `allocatedSize`). Title bytes and device bytes are different numbers and are
  reported separately.
- `create_texture_view()` is the only way two title descriptions of one range
  may share an image when their formats are view-compatible.
- `create_texture_over_memory()` is the zero-copy linear texture path. It is
  valid only for single-mip, single-sample 2D colour formats and
  `minimumLinearTextureAlignment`-aligned offsets and pitches.

### `GraphicsBackend`

Render pipelines and one ordered `PassSubmission` (compute, render, blit).
A pass boundary is an ordering point for title-memory accesses; dropping a
command must not silently join the passes either side of it. The host keeps
that split with its `cut` vector (`native/host.rs`).

### `PresentBackend`

The drawable layer and presentation. Presentation never waits for the GPU.

## Memory model

1. **Guest memory is zero copy.** A pool is one host buffer over the guest
   pages. There is no shadow copy and no copy-back in the native host.
2. **A host texture is a device copy of a guest texture**, because the console
   layout is tiled and Metal's is not. Sampled textures are copied from guest
   memory; render targets and UAV textures live only on the device after the
   GPU writes them.
3. **Backing identity is the guest range plus the console layout**, not the
   resource object. Two title resources that name the same range and layout
   share one image. Exact-range sharing is implemented; format-view sharing
   and true range-overlap sharing are the next steps.
4. **GPU ownership is explicit.** `gpu_written` is set by a target bind, a
   clear, a copy destination or a UAV. A GPU-written texture may not be given
   up until its contents are downloaded or the title proves it no longer needs
   them.
5. **Descriptor entries are repointable.** A heap entry names a title key and
   either the real texture or a typed 1×1 spare. Lazy realization happens at
   encode time for every heap the list can index; dynamic indexing is therefore
   valid even after an eviction. The title record, not the host entry, is the
   source of truth, and its generation rejects reused slots.
6. **Eviction is safe only with a completion gate.** A texture must be
   unbound and older than the last completed submission. Metal retains
   resources named by a committed command buffer, but the host must not reuse
   or download their bytes while work may still write them. Repointing a
   descriptor heap entry additionally waits for the heap's last submission
   serial to complete.
7. **Budgets are backend working-set budgets.** The title's 13.5 GiB guest
   reservation and the host's device allocations compete for the same Apple
   Silicon memory. A host texture budget that ignores
   `recommendedMaxWorkingSetSize` is not a memory policy.
8. **Storage mode is not a memory saving by itself.** A 256 MiB synthetic
   measurement showed `Shared` and `Private` within 0.1 MiB of the same
   physical footprint and device allocation. Treat private storage as an
   access/compression policy, not as duplication removal.

## Upstream decisions

| Upstream | Adopted | Rejected / deferred | Reason |
|---|---|---|---|
| Xenia `shared_memory` | page valid/GPU-written split, one-shot watches, coalesced upload runs | EDRAM ownership/transfers | our targets are Metal textures, not guest-layout EDRAM |
| Xenia `texture_cache` | exact key + LRU, completion gate, mip/slice extents | exact-key-only as the sole identity | overlapping ranges and views still duplicate |
| shadPS4 `texture_cache` | page-bucket range index, view arithmetic, dirty masks, download-before-free under pressure | exact-base-only `FindImageFromRange` | it does not find true overlaps |
| VIctory #2286/#2162/#2455 | one host texture per texture memory, alias groups, depth twins | code copy | D3D12 host; port invariants |
| VIctory descriptor host | repoint viewing slots to null, remirror on use, retire until submission | direct slot tagging | different descriptor API |
| XCL `console_heap.rs` | descriptors are values, slot copy renumbers, lazy shared-heap pages | D3D12 heap plumbing | Metal argument buffers hold resource IDs |
| Xenia `d3d12_command_processor` | two-level index → slot, fixed null stand-ins, completion-gated recycling | raw D3D12 code | portable design |
| DXMT/Mullion descriptor heaps | `{resource_id, metadata}` records, CPU slot table, typed nulls | Wine/D3D12 thunks | portable layout/invariants |
| VIPort `alias_source`/`alias_copy` | copy from a GPU-written alias when formats cannot share | direct code | different backend |
| XCL `gpu/tiling.rs` | GFX10 tiling and format conformance | runtime/Vulkan path | pure Rust tests are reusable |
| Leonida Xbox-ABI proxy | slot/ABI conformance vectors | runtime | Windows/D3D12/Vulkan, does not boot |
| DXMT/Mullion | placement heaps, allocation-info padding, private storage, working-set ceiling, completion thread, watchdog flight dump | global fence for every encoder, sparse/tier-3 until needed, Win32 plumbing | Metal-portable; some are concepts only |

## Rules for integrating a fix

1. Search the upstream repository, forks, branches and PRs first. Record the
   URL/commit in this document or in `experiments/README.md`.
2. Port the implementation, not just the idea, when the languages and
   backends match. If they do not, state the invariant and write a test.
3. Add a synthetic test at the seam. A fix that only works in a full game run
   is not adopted.
4. Measure the same workload before and after: title bytes, device allocation,
   unique backing storage, physical footprint, swap and the frame result.
5. Do not weaken a guard, lower resolution or quality, or fake completion to
   make a run pass.
