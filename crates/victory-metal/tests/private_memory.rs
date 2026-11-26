//! Does `MTLStorageModePrivate` reduce the memory the system actually charges
//! for a render-target-sized allocation, or only change where it lives?
//!
//! Run one mode per process so allocator reuse cannot hide the difference:
//!   cargo test --release -p victory-metal --test private_memory -- --ignored --nocapture shared_256m
//!   cargo test --release -p victory-metal --test private_memory -- --ignored --nocapture private_256m

#![cfg(target_os = "macos")]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::*;
use std::time::Instant;
use victory_platform::mem::{footprint, swap_used};

const SIDE: usize = 1024;
const COUNT: usize = 64;

fn mb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn measure(storage: MTLStorageMode) {
    let device = MTLCreateSystemDefaultDevice().expect("Metal device");
    let queue = device.newCommandQueue().expect("command queue");
    let recommended = device.recommendedMaxWorkingSetSize();
    let before = (
        device.currentAllocatedSize() as u64,
        footprint() as u64,
        swap_used() as u64,
    );

    let mut textures: Vec<Retained<ProtocolObject<dyn MTLTexture>>> = Vec::with_capacity(COUNT);
    for _ in 0..COUNT {
        let td = MTLTextureDescriptor::new();
        td.setTextureType(MTLTextureType::Type2D);
        td.setPixelFormat(MTLPixelFormat::RGBA8Unorm);
        // SAFETY: plain descriptor properties; Metal validates them at creation.
        unsafe {
            td.setWidth(SIDE);
            td.setHeight(SIDE);
        }
        td.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        td.setStorageMode(storage);
        textures.push(device.newTextureWithDescriptor(&td).expect("texture"));
    }

    // A clear pass is enough to prove the allocation is usable and, for private
    // storage, makes the driver back it with physical pages.
    let cb = queue.commandBuffer().expect("command buffer");
    for tex in &textures {
        let pass = MTLRenderPassDescriptor::new();
        // SAFETY: index 0 exists on a fresh render-pass descriptor.
        let a = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
        a.setTexture(Some(tex));
        a.setLoadAction(MTLLoadAction::Clear);
        a.setStoreAction(MTLStoreAction::Store);
        a.setClearColor(MTLClearColor {
            red: 0.0,
            green: 0.0,
            blue: 0.0,
            alpha: 1.0,
        });
        let enc = cb.renderCommandEncoderWithDescriptor(&pass).expect("encoder");
        enc.endEncoding();
    }
    cb.commit();
    cb.waitUntilCompleted();
    if let Some(e) = cb.error() {
        eprintln!("command buffer error: {e:?}");
    }
    // Give the driver a moment to account the backing.
    std::thread::sleep(std::time::Duration::from_millis(250));

    let during = (
        device.currentAllocatedSize() as u64,
        footprint() as u64,
        swap_used() as u64,
    );
    println!(
        "storage={storage:?} textures={COUNT} bytes={} MiB",
        mb((SIDE * SIDE * 4 * COUNT) as u64)
    );
    println!(
        "  currentAllocated {} -> {} MiB (+{:.1})",
        mb(before.0),
        mb(during.0),
        mb(during.0.saturating_sub(before.0))
    );
    println!(
        "  phys_footprint   {} -> {} MiB (+{:.1})",
        mb(before.1),
        mb(during.1),
        mb(during.1.saturating_sub(before.1))
    );
    println!(
        "  system_swap      {} -> {} MiB (+{:.1})",
        mb(before.2),
        mb(during.2),
        mb(during.2.saturating_sub(before.2))
    );
    println!("  recommended      {:.1} MiB", mb(recommended));

    drop(textures);
    let _ = queue;
    std::thread::sleep(std::time::Duration::from_millis(500));
    let after = (
        MTLCreateSystemDefaultDevice()
            .map(|d| d.currentAllocatedSize() as u64)
            .unwrap_or(0),
        footprint() as u64,
        swap_used() as u64,
    );
    println!(
        "  after drop       current +{:.1} MiB, footprint +{:.1} MiB, swap +{:.1} MiB",
        mb(after.0.saturating_sub(before.0)),
        mb(after.1.saturating_sub(before.1)),
        mb(after.2.saturating_sub(before.2))
    );
    let _ = Instant::now();
}

#[test]
#[ignore = "memory experiment; run one mode per process"]
fn shared_256m() {
    measure(MTLStorageMode::Shared);
}

#[test]
#[ignore = "memory experiment; run one mode per process"]
fn private_256m() {
    measure(MTLStorageMode::Private);
}

/// Does wrapping a lazily committed guest pool as a Metal buffer commit or
/// touch its pages? Separates virtual reservation, physical residency and
/// Metal allocation.
#[test]
#[ignore = "memory experiment; needs the Metal Shader Converter"]
fn guest_backing_256m() {
    use victory_backend::backend::ComputeBackend;
    use victory_platform::mem::{PagePool, resident};

    let pool = PagePool::new(256 << 20).expect("guest pool");
    let base = pool.base().expect("pool mapping");
    let before = (
        resident(base as usize, pool.len()),
        footprint() as u64,
        swap_used() as u64,
    );
    let metal = victory_metal::metal::Metal::new().expect("Metal");
    let buf = metal
        .wrap_title_memory(base, pool.len())
        .expect("wrap guest pool");
    std::thread::sleep(std::time::Duration::from_millis(250));
    let wrapped = (
        resident(base as usize, pool.len()),
        footprint() as u64,
        swap_used() as u64,
        metal.memory_usage(),
    );
    // Touch one 16 KiB page after wrapping: resident must grow by that page,
    // not by the whole pool, unless Metal had already committed it.
    // SAFETY: inside the pool mapping.
    unsafe { base.write(1) };
    std::thread::sleep(std::time::Duration::from_millis(100));
    let touched = resident(base as usize, pool.len());
    println!("guest pool {} MiB", mb(pool.len() as u64));
    println!(
        "  resident {} -> {} -> {} MiB",
        mb(before.0 as u64),
        mb(wrapped.0 as u64),
        mb(touched as u64)
    );
    println!(
        "  footprint {} -> {} MiB (+{:.1})",
        mb(before.1),
        mb(wrapped.1),
        mb(wrapped.1.saturating_sub(before.1))
    );
    println!(
        "  swap {} -> {} MiB (+{:.1})",
        mb(before.2),
        mb(wrapped.2),
        mb(wrapped.2.saturating_sub(before.2))
    );
    println!(
        "  metal allocated {} MiB, title buffers {} MiB",
        mb(wrapped.3.allocated),
        mb(wrapped.3.title_buffers)
    );
    drop(buf);
    drop(metal);
    drop(pool);
}
