//! Title shaders on the native host: the console container translated to DXIL for a 32-lane
//! host (`Options { wave32_host: true, .. }`: the console's wave64 compute programs run as two host
//! waves), DXIL to a Metal library through the converter (`create_compute_pipeline` and
//! `create_render_pipeline` on the backend: every library is its own file name, because Metal
//! keeps libraries by URL), pipelines cached by the content of their DXIL, and what the host
//! cannot run refused, logged and counted instead of failing the submission.
//!
//! Title memory at addresses the shader computes is found through the page table of
//! `victory_shader`'s "Title memory", which `super::addr` fills with views of the pools
//! themselves (the pool-offset contract). What a translated shader needs
//! from the host that this one does not give yet, and is refused for: buffer views read from the
//! heap by index, the write marks and view addresses of the shadow-copy design, and the page
//! table for graphics pipelines outside the guessed path. See docs/metal-backend.md.

use super::list::{Binding, ComputePso};
use super::state::{self, Graphics};
use super::title::{Pipe, Title};
use std::collections::HashMap;
use std::rc::Rc;
use victory_backend::resources::*;
use victory_shader::emit::{HostShader, Options};

/// The translation settings of the native host.
pub fn options() -> Options {
    Options {
        wave32_host: true,
        zero_copy_memory: true,
        ..Options::default()
    }
}

/// A console shader container as DXIL for the native host.
pub fn translate(container: &[u8]) -> Result<HostShader, String> {
    victory_shader::emit::translate_with(container, options()).map_err(|e| e.to_string())
}

/// A guessed vertex shader's container translated to draw rectangle lists (`Options::rect_list`).
pub fn translate_rect(container: &[u8]) -> Result<HostShader, String> {
    victory_shader::emit::translate_with(container, Options { rect_list: true, ..options() }).map_err(|e| e.to_string())
}

/// A guessed vertex shader's container translated for rectangle lists or not, writing the
/// `TEXCOORD` parameters in `extra` (bit n) too: those its pixel shader reads and it does not write.
pub fn translate_extra(container: &[u8], rect: bool, extra: u32, copy_params: [u8; 32]) -> Result<HostShader, String> {
    victory_shader::emit::translate_with(container, Options { rect_list: rect, extra_params: extra, copy_params, ..options() })
        .map_err(|e| e.to_string())
}

/// What the host cannot give a translated shader (the reason is the refusal's); `memory`: it can
/// give the page table (compute and guessed graphics shaders).
pub fn unsupported(shader: &HostShader, memory: bool) -> Option<&'static str> {
    if shader.memory_writes {
        return Some("marks the pages it writes (translated without zero_copy_memory)");
    }
    if shader.memory && !memory {
        return Some("reads title memory through the page table in a graphics pipeline");
    }
    if shader.view_addresses || shader.view_marks {
        return Some("needs view addresses or write marks");
    }
    if shader.heap {
        return Some("indexes the descriptor heap itself");
    }
    None
}

/// Why a vertex shader and pixel shader cannot be a pipeline of the host, if they cannot.
fn refusal(vs: &HostShader, ps: Option<&HostShader>, memory: bool) -> Option<&'static str> {
    for s in [Some(vs), ps].into_iter().flatten() {
        if let Some(why) = unsupported(s, memory) {
            return Some(why);
        }
    }
    None
}

/// The thread-group size a compute container declares (`PSV0`'s runtime info).
pub fn threads_of(dxil: &[u8]) -> Option<[usize; 3]> {
    use victory_shader::{Container, FourCC};
    let container = Container::parse(dxil).ok()?;
    let part = container.part(FourCC::PSV0)?;
    // The runtime info is 48 bytes from version 2 on; the group size is its last three words.
    if u32::from_le_bytes(part.data.get(..4)?.try_into().ok()?) < 48 {
        return None;
    }
    let d = part.data.get(40..52)?;
    let n = |k: usize| u32::from_le_bytes(d[k * 4..k * 4 + 4].try_into().unwrap()) as usize;
    Some([n(0), n(1), n(2)]).filter(|t| t.iter().all(|&n| n > 0))
}

/// Counts of what the shader path did, and why it refused what it did.
#[derive(Default, Debug, Clone)]
pub struct ShaderStats {
    /// Pipelines made.
    pub pipelines: u64,
    /// Pipelines served from the library cache.
    pub cached: u64,
    /// Refusals, by reason, with how often.
    pub refused: Vec<(String, u64)>,
}

impl ShaderStats {
    /// All refusals.
    pub fn refusals(&self) -> u64 {
        self.refused.iter().map(|r| r.1).sum()
    }

    fn refuse(&mut self, what: &str, reason: String) {
        let key = format!("{what}: {reason}");
        match self.refused.iter_mut().find(|r| r.0 == key) {
            Some(r) => r.1 += 1,
            None => {
                tracing::warn!(shader = what, %reason, "title shader refused; its work is dropped");
                if self.refused.len() < 256 {
                    self.refused.push((key, 1));
                }
            }
        }
    }
}

/// The pipelines made, by the content of their DXIL (and, for graphics, their state).
pub struct Libraries<B: GraphicsBackend> {
    compute: HashMap<[u8; 32], Pipe<B>>,
    render: HashMap<[u8; 32], Pipe<B>>,
}

impl<B: GraphicsBackend> Default for Libraries<B> {
    fn default() -> Self {
        Libraries {
            compute: HashMap::new(),
            render: HashMap::new(),
        }
    }
}

impl<B: PresentBackend> Title<B> {
    /// The compute pipeline of a translated shader: from the cache when its DXIL was converted
    /// before, else converted now. [`Pipe::Refused`] (counted) when the host cannot run it.
    pub fn compute_pipe(&mut self, shader: &HostShader, name: &str) -> Pipe<B> {
        if let Some(why) = unsupported(shader, true) {
            self.shaders.refuse(name, why.into());
            return Pipe::Refused;
        }
        let key = *blake3::hash(&shader.dxil).as_bytes();
        if let Some(p) = self.libraries.compute.get(&key) {
            self.shaders.cached += 1;
            return p.clone();
        }
        let made = self.make_compute(shader, name);
        self.libraries.compute.insert(key, made.clone());
        made
    }

    fn make_compute(&mut self, shader: &HostShader, name: &str) -> Pipe<B> {
        let Some(threads) = threads_of(&shader.dxil) else {
            self.shaders.refuse(name, "no thread group size".into());
            return Pipe::Refused;
        };
        // Translator output carries no root signature: the argument buffer is laid out by the
        // binding the host fills it by.
        let binding = Binding::of(shader);
        let started = std::time::Instant::now();
        let made = match binding.layout() {
            Some(l) => self.host.backend.create_compute_pipeline_with(&shader.dxil, &l),
            None => self.host.backend.create_compute_pipeline(&shader.dxil),
        };
        let elapsed = started.elapsed();
        if std::env::var_os("VICTORY_TRACE_PIPELINE_TIME").is_some() {
            tracing::info!(name, ms = elapsed.as_millis() as u64, "compute pipeline made");
        }
        match made {
            Ok(pso) => {
                self.shaders.pipelines += 1;
                Pipe::Compute(Rc::new(ComputePso {
                    pso,
                    threads,
                    binding: Some(binding),
                }))
            }
            Err(e) => {
                self.shaders.refuse(name, format!("conversion: {e}"));
                Pipe::Refused
            }
        }
    }

    /// The render pipeline of a translated vertex shader, pixel shader and graphics state.
    pub fn render_pipe(
        &mut self,
        vs: &HostShader,
        ps: Option<&HostShader>,
        graphics: &Graphics,
        name: &str,
    ) -> Pipe<B> {
        if let Some(why) = refusal(vs, ps, graphics.is_guessed()) {
            self.shaders.refuse(name, why.into());
            return Pipe::Refused;
        }
        let mut h = blake3::Hasher::new();
        h.update(&vs.dxil);
        h.update(ps.map_or(&[][..], |p| &p.dxil));
        h.update(format!("{graphics:?}").as_bytes());
        // The arguments are part of the pipeline's binding.
        h.update(format!("{:?}{:?}", vs.user_data, ps.map(|p| &p.user_data)).as_bytes());
        let key = *h.finalize().as_bytes();
        if let Some(p) = self.libraries.render.get(&key) {
            self.shaders.cached += 1;
            return p.clone();
        }
        let made = self.make_render(vs, ps, graphics, name);
        self.libraries.render.insert(key, made.clone());
        made
    }

    fn make_render(
        &mut self,
        vs: &HostShader,
        ps: Option<&HostShader>,
        g: &Graphics,
        name: &str,
    ) -> Pipe<B> {
        let Some(reads) = state::vertex_inputs(&vs.dxil) else {
            self.shaders.refuse(name, "no input signature".into());
            return Pipe::Refused;
        };
        let Some((mut attributes, streams)) = g.vertex_layout(&reads) else {
            self.shaders
                .refuse(name, "the vertex shader reads an input the layout lacks".into());
            return Pipe::Refused;
        };
        // A system value ahead of the attributes takes an attribute number: described as the
        // attribute after it, which the converter's function ignores.
        for at in state::system_gaps(&vs.dxil) {
            if let Some(next) = attributes.get(at).copied() {
                attributes.insert(at, next);
            }
        }
        let (Some(color), Ok(depth)) = (g.color_formats(), g.depth_format()) else {
            self.shaders.refuse(name, "a target format the host does not know".into());
            return Pipe::Refused;
        };
        if g.samples > 1 {
            self.shaders.refuse(name, "multisampled targets".into());
            return Pipe::Refused;
        }
        let color_state = g.color_state();
        if std::env::var_os("VICTORY_TRACE_PIPES").is_some() {
            tracing::info!(name, stencil = ?g.stencil(), depth = ?g.depth_test(), ?color_state, "guessed pipeline state");
        }
        let vs_binding = Binding::of(vs);
        let ps_binding = ps
            .map(Binding::of)
            .filter(|b| b.memory || b.dwords > 0 || !b.views.is_empty());
        // As for compute: without the host's layout the converter makes the user-data block a
        // root CBV (a pointer), while the host writes its dwords inline, so the shader reads its
        // user data (descriptor table offsets, root constants) from address 0.
        let vs_layout = vs_binding.layout();
        let ps_layout = ps.map(Binding::of).and_then(|b| b.layout());
        let desc = RenderPipelineDesc {
            vs: &vs.dxil,
            ps: ps.map(|p| &p.dxil[..]),
            vs_layout: vs_layout.as_ref(),
            ps_layout: ps_layout.as_ref(),
            attributes: &attributes,
            streams: &streams,
            color: &color,
            depth,
            depth_test: depth.and(g.depth_test()),
            cull: g.cull(),
            color_state: &color_state,
            stencil: depth.and(g.stencil()),
            raster: g.raster(),
        };
        let started = std::time::Instant::now();
        let made = self.host.render_pipeline(&desc);
        let elapsed = started.elapsed();
        if std::env::var_os("VICTORY_TRACE_PIPELINE_TIME").is_some() {
            tracing::info!(name, ms = elapsed.as_millis() as u64, "render pipeline made");
        }
        match made {
            Ok(p) => {
                self.shaders.pipelines += 1;
                Pipe::Render(p, Some(Rc::new(Binding {
                    ps: ps_binding.map(Rc::new),
                    ..vs_binding
                })))
            }
            Err(e) => {
                self.shaders.refuse(name, format!("conversion: {e}"));
                Pipe::Refused
            }
        }
    }

    /// A translation that failed, counted.
    pub fn refused_translation(&mut self, name: &str, reason: String) {
        self.shaders.refuse(name, format!("translation: {reason}"));
    }
}
