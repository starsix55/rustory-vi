//! Hand-assembled GFX10 unit tests covering all 6 graphics shader refusal
//! classes (both positive sequences that must lower — across `wave32_host`
//! false/true where relevant — and genuinely unsupported variants that must
//! remain refused). Encodings assembled with `llvm-mc -mcpu=gfx1030` (plus
//! `-mattr=+wavefrontsize64` for wave64).

use super::container::SigElement;
use super::lower::{self, Lowered, Parts};
use super::{Options, graphics, translate_with};
use crate::dxbc::{Container, FourCC};
use crate::native::{HwStage, NativeProgram, PsInput, RootParam, RootType, StageIo, Usage};
use crate::{ShaderError, io, ir, resource, structure};

fn bytes_of(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

#[test]
fn guessed_pixel_computed_buffer_loads_use_title_memory() {
    let mut meta = [0u8; 0x30];
    meta[12..15].copy_from_slice(&[0x40, 0, 1]);
    meta[0x10] = 0x41;
    meta[0x14] = 2; // CBV, root constants in slot 0, live-in s0
    let code = [
        0x8004_9000, // s_add_u32 s4, s0, 16
        0x8205_8080, // s_addc_u32 s5, 0, 0
        0x8705_FF05,
        0x0000_FFFF, // s_and_b32 s5, s5, 0xffff
        0x8F06_9090, // s_lshl_b32 s6, 16, 16
        0x8805_0605, // s_or_b32 s5, s5, s6
        0xBE86_0384, // s_mov_b32 s6, 4
        0xBE87_03FF,
        0x0001_4FAC, // s_mov_b32 s7, 0x00014fac
        0xE034_3004,
        0x8001_0200, // buffer_load_dwordx2 v[2:3], v[0:1], s[4:7], 0 idxen offen offset:4
        0xF420_0202,
        0xFA00_000C, // s_buffer_load_dword s8, s[4:7], 12
        0x7E08_0208, // v_mov_b32 v4, s8
        0xF800_180F,
        0x0404_0302, // exp mrt0 v2, v3, v4, v4 done vm
        0xBF81_0000,
    ];
    let container = |code: &[u32]| graphics::container(HwStage::Ps, &bytes_of(code), &meta).unwrap();
    let options = Options { wave32_host: true, zero_copy_memory: true, ..Options::default() };
    for flags in [0x0001_4FAC, 0x3001_4FAC] {
        let mut code = code;
        code[8] = flags;
        let shader = translate_with(&container(&code), options).unwrap();
        assert!(shader.memory);
        assert!(!shader.memory_writes && !shader.heap);
        assert_eq!(shader.user_data.len(), 1);
        assert_eq!(shader.user_data[0].slot, 0);
    }
    let mut dynamic_count = code;
    dynamic_count[6] = 0xBE86_0300; // s_mov_b32 s6, s0
    assert!(translate_with(&container(&dynamic_count), options).unwrap().memory);
    assert!(translate_with(&container(&code), Options::default()).is_err());

    let mut swizzled = code;
    swizzled[5] = 0x8805_FF05; // s_or_b32 s5, s5, 0x80000000
    let mut swizzled = swizzled.to_vec();
    swizzled.insert(6, 0x8000_0000);
    let err = translate_with(&container(&swizzled), options).err().unwrap();
    assert!(err.to_string().contains("computed buffer descriptor with swizzling"), "{err}");
}

#[test]
fn guessed_pixel_computed_subword_loads_use_title_memory() {
    let mut meta = [0u8; 0x30];
    meta[12..15].copy_from_slice(&[0x40, 0, 1]);
    meta[0x10] = 0x41;
    meta[0x14] = 2;
    let code = [
        0x8004_9000, // s_add_u32 s4, s0, 16
        0x8205_8080, // s_addc_u32 s5, 0, 0
        0x8705_FF05, 0x0000_FFFF, // s_and_b32 s5, s5, 0xffff
        0x8F06_9090, // s_lshl_b32 s6, 16, 16
        0x8805_0605, // s_or_b32 s5, s5, s6
        0xBE86_0384, // s_mov_b32 s6, 4
        0xBE87_03FF, 0x3001_4FAC, // s_mov_b32 s7, flags
        0xE020_3000, 0x8001_0200, // buffer_load_ubyte v2, v[0:1], s[4:7], 0 idxen offen
        0xF800_180F, 0x0202_0202, // exp mrt0 v2, v2, v2, v2 done vm
        0xBF81_0000,
    ];
    let translate = |words: &[u32], wave32_host| {
        let container = graphics::container(HwStage::Ps, &bytes_of(words), &meta).unwrap();
        translate_with(&container, Options { wave32_host, zero_copy_memory: true, ..Options::default() })
    };
    for wave32_host in [false, true] {
        for op in 8..=11 {
            for bounds in 0..4 {
                for offset in [0, 1, 2, 3, 15, 16, 4095] {
                    let mut words = code;
                    words[8] = 0x0001_4FAC | bounds << 28;
                    words[9] = 0xE000_3000 | op << 18 | offset;
                    let shader = translate(&words, wave32_host).unwrap();
                    assert!(shader.memory && !shader.memory_writes && !shader.heap);
                }
            }
        }
    }
    // Typed loads, stores and atomics must not gain support accidentally.
    for op in [0, 24, 26, 28, 50] {
        let mut words = code;
        words[9] = 0xE000_3000 | op << 18;
        assert!(translate(&words, true).is_err(), "opcode {op}");
    }
    for flags in [0x3081_4FAC, 0x7001_4FAC] {
        let mut words = code;
        words[8] = flags; // ADD_TID or non-buffer type
        assert!(translate(&words, true).is_err());
    }
    let mut swizzled = code.to_vec();
    swizzled[5] = 0x8805_FF05;
    swizzled.insert(6, 0x8000_0000);
    assert!(translate(&swizzled, true).is_err());
}

/// Exercise the graphics descriptor classification with a compute entry point
/// so the Windows test host can dispatch it and read back exact integer results.
/// This does not enable computed descriptors for production compute shaders.
pub(super) fn computed_subword_test_shader(op: u32, bounds: u32) -> super::HostShader {
    use crate::native::{ComputeIo, ResourceType, UsedResource};
    let words = [
        0xBE88_0300, // s_mov_b32 s8, s0 (base low)
        0x8709_FF01, 0x3FFF_FFFF, // s_and_b32 s9, s1, 0x3fffffff (base high + stride)
        0xBE8A_0302, // s_mov_b32 s10, s2 (record count)
        0xBE8B_03FF, 0x0001_4FAC | bounds << 28,
        0x7E00_0203, // v_mov_b32 v0, s3 (index)
        0x7E02_0204, // v_mov_b32 v1, s4 (vector offset)
        0xE000_3000 | op << 18, 0x0502_0200, // buffer_load_* v2, v[0:1], s[8:11], s5 idxen offen
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF408_0303, 0xFA00_0000, // s_load_dwordx4 s[12:15], s[6:7], 0
        0xBF8C_0000, // s_waitcnt
        0xE070_0000, 0x8003_0200, // buffer_store_dword v2, off, s[12:15], 0
        0xBF81_0000,
    ];
    let native = NativeProgram {
        stage: HwStage::Cs, wave64: false, vgprs: 4,
        layout: vec![root_const(0, 6), RootParam {
            root: RootType::Table, table_offset_dwords: 0, sgprs: 1,
            base_register: 0, user_data: 6, slot: 1,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav, root: RootType::Table, table_offset_dwords: 0,
            lower_bound: 0, stride: 0, dwords: 4, user_data: 6, slot: 1,
        }],
        io: StageIo::Cs(ComputeIo { rsrc2: 7 << 1, threads: [1, 1, 1] }),
        code: bytes_of(&words),
    };
    let p = ir::build(&native.code, false, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track_graphics(&p, &native.layout, HwStage::Ps, Some(&st), true).unwrap();
    assert!(res.descriptors.iter().any(|d| matches!(d.source, resource::Source::BuiltMemory { .. })));
    let io = io::raise(&p, &native).unwrap();
    let l = lower::lower(&Parts {
        native: &native, p: &p, st: &st, io: &io, res: &res,
        inputs: vec![], outputs: vec![], prims: vec![], mesh_stride: 0,
        atomic_locks: false, bias_as_gradients: false, narrow_waves: false,
        wave32_host: false, zero_copy_memory: true, view_marks: false,
        rect_list: false, copy_params: [0; 32],
    }).unwrap();
    super::HostShader {
        stage: HwStage::Cs, dxil: l.dxil, splits: l.splits, table_entries: l.table_entries,
        table_dynamic: l.table_dynamic,
        user_data: l.user_data, samplers: l.samplers, static_samplers: vec![],
        root_cbvs: l.root_cbvs, root_srvs: l.root_srvs, root_uavs: l.root_uavs,
        memory: l.memory, memory_writes: l.memory_writes, view_addresses: l.view_addresses,
        heap: l.heap, heap_writes: l.heap_writes, view_marks: l.view_marks,
    }
}

#[test]
fn computed_subword_execution_fixtures_lower() {
    for op in 8..=11 {
        for bounds in 0..4 {
            assert!(computed_subword_test_shader(op, bounds).memory);
        }
    }
}

fn sig_el(
    name: &str,
    index: u32,
    kind: u8,
    comp: u8,
    interp: u8,
    row: Option<u32>,
    cols: u8,
) -> SigElement {
    SigElement {
        name: name.into(),
        index,
        kind,
        comp,
        interp,
        row,
        col: 0,
        cols,
        usage: 0,
        half: false,
    }
}

#[allow(clippy::too_many_arguments)]
fn lower_ps(
    words: &[u32],
    wave64: bool,
    layout: Vec<RootParam>,
    ps_inputs: Vec<PsInput>,
    input_addr: u32,
    db_shader_control: u32,
    inputs: Vec<SigElement>,
    wave32_host: bool,
) -> Result<Lowered, ShaderError> {
    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64,
        vgprs: 16,
        layout,
        used: vec![],
        io: StageIo::Ps {
            inputs: ps_inputs,
            input_addr,
            db_shader_control,
            patches: vec![],
        },
        code: bytes_of(words),
    };
    let p = ir::build(&native.code, wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
    lower::lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs,
        outputs: vec![sig_el("SV_Target", 0, 16, 9, 0, Some(0), 4)],
        prims: vec![],
        mesh_stride: 0,
        atomic_locks: false,
        bias_as_gradients: false,
        narrow_waves: false,
        wave32_host,
        zero_copy_memory: false,
        view_marks: false,
        rect_list: false,
        copy_params: [0; 32],
    })
}

fn lower_vs(
    words: &[u32],
    wave64: bool,
    layout: Vec<RootParam>,
    rect_list: bool,
) -> Result<Lowered, ShaderError> {
    let native = NativeProgram {
        stage: HwStage::Vs,
        wave64,
        vgprs: 16,
        layout,
        used: vec![],
        io: StageIo::Vs {
            inputs: vec![],
            outputs: vec![],
        },
        code: bytes_of(words),
    };
    let p = ir::build(&native.code, wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
    lower::lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![sig_el("SV_Position", 0, 3, 9, 0, Some(0), 4)],
        prims: vec![],
        mesh_stride: 0,
        atomic_locks: false,
        bias_as_gradients: false,
        narrow_waves: false,
        wave32_host: false,
        zero_copy_memory: false,
        view_marks: false,
        rect_list,
        copy_params: [0; 32],
    })
}

fn parsed_isg1(dxil: &[u8]) -> Vec<SigElement> {
    let c = Container::parse(dxil).unwrap();
    let part = c.part(FourCC::ISG1).expect("ISG1");
    super::signature(part.data).unwrap()
}

fn root_const(user_data: u8, sgprs: u32) -> RootParam {
    RootParam {
        root: RootType::Constants,
        table_offset_dwords: 0,
        sgprs,
        base_register: 0,
        user_data,
        slot: 0,
    }
}

fn vertex_table_layout() -> Vec<RootParam> {
    vec![RootParam {
        root: RootType::Descriptor,
        table_offset_dwords: 0,
        sgprs: 2,
        base_register: 0,
        user_data: 0,
        slot: graphics::VERTEX_TABLE,
    }]
}

// =========================================================================
// Refusal #1: "opaque lane mask" / "a constant lane mask" in MaskData::Host
// =========================================================================

#[test]
fn refusal1_mask_scc_lowers_in_pixel_shader() {
    // v_cmp_lt_u32_e32 vcc, 5, v0
    // s_and_b64 s[2:3], vcc, exec
    // s_cselect_b32 s4, 7, 9
    // v_mov_b32_e32 v0, s4
    // exp mrt0 v0, v0, v0, v0 done vm
    // s_endpgm
    let code = [
        0x7D82_0085,
        0x8782_7E6A,
        0x8504_8987,
        0x7E00_0204,
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    for wave32_host in [false, true] {
        let out = lower_ps(&code, true, vec![], vec![], 0, 0, vec![], wave32_host)
            .unwrap_or_else(|e| panic!("wave32_host={wave32_host}: {e}"));
        assert!(out.wave_ops);
    }
}

#[test]
fn refusal1_wave_is_first_lane_idioms_lower_in_pixel_shader() {
    // Wave64: 1 << s_ff1_i32_b64(exec)
    let w64_lshl = [
        0xBE80_147E, // s_ff1_i32_b64 s0, exec
        0x8F82_0081, // s_lshl_b64 s[2:3], 1, s0
        0xBE84_2402, // s_and_saveexec_b64 s[4:5], s[2:3]
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        0xF800_180F,
        0x0000_0000, // exp mrt0 v0, v0, v0, v0 done vm
        0xBF81_0000, // s_endpgm
    ];
    // Wave64: s_bitset1_b64(0, s_ff1_i32_b64(exec))
    let w64_bitset1 = [
        0xBE82_0480, // s_mov_b64 s[2:3], 0
        0xBE80_147E, // s_ff1_i32_b64 s0, exec
        0xBE82_1E00, // s_bitset1_b64 s[2:3], s0
        0xBE84_2402, // s_and_saveexec_b64 s[4:5], s[2:3]
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // Wave64: 1 << (63 - s_flbit_i32_b64(exec))
    let w64_flbit = [
        0xBE80_167E, // s_flbit_i32_b64 s0, exec
        0x8180_00BF, // s_sub_i32 s0, 63, s0
        0x8F82_0081, // s_lshl_b64 s[2:3], 1, s0
        0xBE84_2402, // s_and_saveexec_b64 s[4:5], s[2:3]
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // Wave32: 1 << s_ff1_i32_b32(exec_lo)
    let w32_lshl = [
        0xBE80_137E, // s_ff1_i32_b32 s0, exec_lo
        0x8F02_0081, // s_lshl_b32 s2, 1, s0
        0xBE84_3C02, // s_and_saveexec_b32 s4, s2
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0304, // s_mov_b32 exec_lo, s4
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // Wave32: 1 << (31 - s_flbit_i32_b32(exec_lo))
    let w32_flbit = [
        0xBE80_157E, // s_flbit_i32_b32 s0, exec_lo
        0x8180_009F, // s_sub_i32 s0, 31, s0
        0x8F02_0081, // s_lshl_b32 s2, 1, s0
        0xBE84_3C02, // s_and_saveexec_b32 s4, s2
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0304, // s_mov_b32 exec_lo, s4
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    for (label, code, wave64) in [
        ("w64_lshl", &w64_lshl[..], true),
        ("w64_bitset1", &w64_bitset1[..], true),
        ("w64_flbit", &w64_flbit[..], true),
        ("w32_lshl", &w32_lshl[..], false),
        ("w32_flbit", &w32_flbit[..], false),
    ] {
        for wave32_host in [false, true] {
            let out = lower_ps(code, wave64, vec![], vec![], 0, 0, vec![], wave32_host)
                .unwrap_or_else(|e| panic!("{label} wave32_host={wave32_host}: {e}"));
            assert!(out.wave_ops, "{label}");
        }
    }
}

#[test]
fn refusal1_mask_bit_tricks_lower_in_pixel_shader() {
    // Wave64: mask & -mask
    let w64_neg = [
        0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
        0x8082_6A80, // s_sub_u32 s2, 0, vcc_lo
        0x8283_6B80, // s_subb_u32 s3, 0, vcc_hi
        0x8782_026A, // s_and_b64 s[2:3], vcc, s[2:3]
        0xBE84_2402, // s_and_saveexec_b64 s[4:5], s[2:3]
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // Wave64: mask & (mask - 1)
    let w64_dec = [
        0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
        0x8082_816A, // s_sub_u32 s2, vcc_lo, 1
        0x8283_806B, // s_subb_u32 s3, vcc_hi, 0
        0x8782_026A, // s_and_b64 s[2:3], vcc, s[2:3]
        0xBE84_2402, // s_and_saveexec_b64 s[4:5], s[2:3]
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // Wave64: s_bitset0_b64(mask, s_ff1_i32_b64(mask))
    let w64_bitset0 = [
        0xBE82_047E, // s_mov_b64 s[2:3], exec
        0xBE80_1402, // s_ff1_i32_b64 s0, s[2:3]
        0xBE82_1C00, // s_bitset0_b64 s[2:3], s0
        0xBE84_2402, // s_and_saveexec_b64 s[4:5], s[2:3]
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // Wave32: mask & -mask
    let w32_neg = [
        0x7D82_0085, // v_cmp_lt_u32_e32 vcc_lo, 5, v0
        0x8182_6A80, // s_sub_i32 s2, 0, vcc_lo
        0x8702_026A, // s_and_b32 s2, vcc_lo, s2
        0xBE84_3C02, // s_and_saveexec_b32 s4, s2
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0304, // s_mov_b32 exec_lo, s4
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // Wave32: mask & (mask - 1)
    let w32_dec = [
        0x7D82_0085, // v_cmp_lt_u32_e32 vcc_lo, 5, v0
        0x8182_816A, // s_sub_i32 s2, vcc_lo, 1
        0x8702_026A, // s_and_b32 s2, vcc_lo, s2
        0xBE84_3C02, // s_and_saveexec_b32 s4, s2
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0304, // s_mov_b32 exec_lo, s4
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    for (label, code, wave64) in [
        ("w64_neg", &w64_neg[..], true),
        ("w64_dec", &w64_dec[..], true),
        ("w64_bitset0", &w64_bitset0[..], true),
        ("w32_neg", &w32_neg[..], false),
        ("w32_dec", &w32_dec[..], false),
    ] {
        for wave32_host in [false, true] {
            lower_ps(code, wave64, vec![], vec![], 0, 0, vec![], wave32_host)
                .unwrap_or_else(|e| panic!("{label} wave32_host={wave32_host}: {e}"));
        }
    }
}

#[test]
fn refusal1_uniform_and_quad_periodic_masks_lower_in_pixel_shader() {
    // Wave32: s_cselect_b32 -1, 0
    let w32_cselect = [
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0x8502_80C1, // s_cselect_b32 s2, -1, 0
        0xBE84_3C02, // s_and_saveexec_b32 s4, s2
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0304, // s_mov_b32 exec_lo, s4
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // Wave32: s_ashr_i32 x, 31
    let w32_ashr = [
        0x9102_9F00, // s_ashr_i32 s2, s0, 31
        0xBE84_3C02, // s_and_saveexec_b32 s4, s2
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0304, // s_mov_b32 exec_lo, s4
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // Wave32: s_sub_i32 0, (s0 & 1)
    let w32_sub_flag = [
        0x8701_8100, // s_and_b32 s1, s0, 1
        0x8182_0180, // s_sub_i32 s2, 0, s1
        0xBE84_3C02, // s_and_saveexec_b32 s4, s2
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0304, // s_mov_b32 exec_lo, s4
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // Wave64: s_cselect_b64 -1, 0
    let w64_cselect = [
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0x8582_80C1, // s_cselect_b64 s[2:3], -1, 0
        0xBE84_2402, // s_and_saveexec_b64 s[4:5], s[2:3]
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // Wave64 quad-periodic constant 0x5555_5555_5555_5555
    let w64_quad = [
        0xBE82_03FF,
        0x5555_5555, // s_mov_b32 s2, 0x55555555
        0xBE83_03FF,
        0x5555_5555, // s_mov_b32 s3, 0x55555555
        0xBE84_2402, // s_and_saveexec_b64 s[4:5], s[2:3]
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // Wave32 quad-periodic constant 0x5555_5555
    let w32_quad = [
        0xBE84_3CFF,
        0x5555_5555, // s_and_saveexec_b32 s4, 0x55555555
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0304, // s_mov_b32 exec_lo, s4
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    for (label, code, wave64) in [
        ("w32_cselect", &w32_cselect[..], false),
        ("w32_ashr", &w32_ashr[..], false),
        ("w32_sub_flag", &w32_sub_flag[..], false),
        ("w64_cselect", &w64_cselect[..], true),
        ("w64_quad", &w64_quad[..], true),
        ("w32_quad", &w32_quad[..], false),
    ] {
        for wave32_host in [false, true] {
            lower_ps(
                code,
                wave64,
                vec![root_const(0, 1)],
                vec![],
                0,
                0,
                vec![],
                wave32_host,
            )
            .unwrap_or_else(|e| panic!("{label} wave32_host={wave32_host}: {e}"));
        }
    }
}

#[test]
fn refusal1_split_exec_and_guessed_wave64_inference() {
    // Split 32-bit moves of EXEC in wave64 PS
    let split_exec = [
        0xBE82_037E, // s_mov_b32 s2, exec_lo
        0xBE83_037F, // s_mov_b32 s3, exec_hi
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0302, // s_mov_b32 exec_lo, s2
        0xBEFF_0303, // s_mov_b32 exec_hi, s3
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    lower_ps(&split_exec, true, vec![], vec![], 0, 0, vec![], true).unwrap();

    // Guessed wave64 PS with v_cmp_lt_u32_e32 (which formats implicit vcc as
    // `vcc_lo` in text(false)): must be inferred as wave64 and lower cleanly.
    let guessed_w64 = [
        0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
        0xBE82_246A, // s_and_saveexec_b64 s[2:3], vcc
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0402, // s_mov_b64 exec, s[2:3]
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    let code = bytes_of(&guessed_w64);
    let container = graphics::container(HwStage::Ps, &code, &[0u8; 0x10]).unwrap();
    let translated = translate_with(
        &container,
        Options {
            wave32_host: true,
            ..Options::default()
        },
    )
    .unwrap();
    assert!(!translated.dxil.is_empty());
}

#[test]
fn refusal1_console_layout_dependent_masks_remain_refused() {
    // 1. Opaque mask from root constants s[0:1]
    let from_root = [
        0xBE84_2400, // s_and_saveexec_b64 s[4:5], s[0:1]
        0x7E00_0281, // v_mov_b32_e32 v0, 1
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // 2. Ballot shifted by non-zero constant: s_lshl_b64 s[2:3], vcc, 1
    let shifted_ballot = [
        0x7D82_0085, // v_cmp_lt_u32_e32 vcc, 5, v0
        0x8F82_816A, // s_lshl_b64 s[2:3], vcc, 1
        0xBE84_2402, // s_and_saveexec_b64 s[4:5], s[2:3]
        0x7E00_0281,
        0xBEFE_0404,
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // 3. Constant single-lane mask: 1 << 5
    let const_bit = [
        0x8F82_8581, // s_lshl_b64 s[2:3], 1, 5
        0xBE84_2402, // s_and_saveexec_b64 s[4:5], s[2:3]
        0x7E00_0281,
        0xBEFE_0404,
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // 4. Non-quad-periodic constant: 0xFFFF_FFFF_0000_0000 (wave64)
    let non_quad_w64 = [
        0xBE82_0380, // s_mov_b32 s2, 0
        0xBE83_03C1, // s_mov_b32 s3, -1
        0xBE84_2402, // s_and_saveexec_b64 s[4:5], s[2:3]
        0x7E00_0281,
        0xBEFE_0404,
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // 5. Non-quad-periodic constant: 0x0000_FFFF (wave32)
    let non_quad_w32 = [
        0xBE84_3CFF,
        0x0000_FFFF, // s_and_saveexec_b32 s4, 0xffff
        0x7E00_0281,
        0xBEFE_0304,
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    // 6. Multiply of a mask: s_mul_i32 s2, vcc_lo, 3 (wave32)
    let mul_mask = [
        0x7D82_0085, // v_cmp_lt_u32_e32 vcc_lo, 5, v0
        0x9302_836A, // s_mul_i32 s2, vcc_lo, 3
        0xBE84_3C02, // s_and_saveexec_b32 s4, s2
        0x7E00_0281,
        0xBEFE_0304,
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    for (label, code, wave64, expect) in [
        ("from_root", &from_root[..], true, "opaque lane mask"),
        ("shifted_ballot", &shifted_ballot[..], true, "opaque lane mask"),
        ("const_bit", &const_bit[..], true, "opaque lane mask"),
        ("non_quad_w64", &non_quad_w64[..], true, "lane mask"),
        ("non_quad_w32", &non_quad_w32[..], false, "constant lane mask"),
        ("mul_mask", &mul_mask[..], false, "opaque lane mask"),
    ] {
        let err = lower_ps(
            code,
            wave64,
            vec![root_const(0, 2)],
            vec![],
            0,
            0,
            vec![],
            true,
        )
        .err()
        .unwrap_or_else(|| panic!("{label}: expected refusal"));
        assert!(
            err.to_string().contains(expect),
            "{label}: expected `{expect}`, got `{err}`"
        );
    }
}

// =========================================================================
// Refusal #2: "a pixel system input the signature lacks"
// =========================================================================

#[test]
fn refusal2_pixel_system_inputs_lower_when_declared() {
    // Reads v0..v6 = Position.xyzw, FrontFace, Ancillary, SampleCoverage
    let code = [
        0x4A00_0300, // v_add_nc_u32_e32 v0, v0, v1
        0x4A00_0500, // v_add_nc_u32_e32 v0, v0, v2
        0x4A00_0700, // v_add_nc_u32_e32 v0, v0, v3
        0x4A00_0900, // v_add_nc_u32_e32 v0, v0, v4
        0x4A00_0B00, // v_add_nc_u32_e32 v0, v0, v5
        0x4A00_0D00, // v_add_nc_u32_e32 v0, v0, v6
        0xF800_180F,
        0x0000_0000, // exp mrt0 v0, v0, v0, v0 done vm
        0xBF81_0000, // s_endpgm
    ];
    let inputs = vec![
        sig_el("SV_Position", 0, 3, 9, 4, Some(0), 4),
        sig_el("SV_IsFrontFace", 0, 13, 5, 1, Some(1), 1),
        sig_el("SV_SampleIndex", 0, 12, 5, 0, None, 1),
        sig_el("SV_Coverage", 0, 14, 5, 0, None, 1),
    ];
    let out = lower_ps(&code, true, vec![], vec![], 0x7F00, 0, inputs, false).unwrap();
    let isg1 = parsed_isg1(&out.dxil);
    assert!(isg1.iter().any(|e| e.kind == 3));
    assert!(isg1.iter().any(|e| e.kind == 13));
    assert!(isg1.iter().any(|e| e.kind == 12));
    assert!(isg1.iter().any(|e| e.kind == 14));

    // Guessed PS (`FSPG`) infers all 4 system values automatically.
    let container = graphics::container(HwStage::Ps, &bytes_of(&code), &[0u8; 0x10]).unwrap();
    let translated = translate_with(&container, Options::default()).unwrap();
    let guessed_isg1 = parsed_isg1(&translated.dxil);
    for kind in [3u8, 12, 13, 14] {
        assert!(
            guessed_isg1.iter().any(|e| e.kind == kind),
            "guessed ISG1 missing system value kind {kind}"
        );
    }
}

#[test]
fn refusal2_missing_pixel_system_inputs_remain_refused() {
    let read_v0 = [
        0xF800_180F,
        0x0000_0000, // exp mrt0 v0, v0, v0, v0 done vm
        0xBF81_0000, // s_endpgm
    ];
    for (label, addr) in [
        ("Position", 1u32 << 8),
        ("FrontFace", 1u32 << 12),
        ("Ancillary", 1u32 << 13),
        ("SampleCoverage", 1u32 << 14),
    ] {
        let err = lower_ps(&read_v0, true, vec![], vec![], addr, 0, vec![], false)
            .err()
            .unwrap_or_else(|| panic!("{label}: expected refusal"));
        assert!(
            err.to_string()
                .contains("a pixel system input the signature lacks"),
            "{label}: got `{err}`"
        );
    }
}

// =========================================================================
// Refusal #3: "a hardware SGPR input"
// =========================================================================

#[test]
fn refusal3_interp_m0_sgpr_and_pops_lower() {
    let ps_in = vec![PsInput {
        usage: Usage::Generic,
        row: 0,
        mask: 1,
        attr: 0,
        flat: false,
        fp16: false,
        pack2: false,
        row2: 0,
    }];
    let sig_in = vec![sig_el("TEXCOORD", 0, 0, 9, 0, Some(0), 4)];

    // 1. Direct `s_mov_b32 m0, s0` (s0 is below default `initialized_sgprs - 2`)
    //    plus `s_sendmsg(MSG_ORDERED_PS_DONE)` which implicitly reads M0 in IR.
    let direct_m0 = [
        0xBEFC_0300, // s_mov_b32 m0, s0
        0xC810_0000, // v_interp_p1_f32 v4, v0, attr0.x
        0xC811_0001, // v_interp_p2_f32 v4, v1, attr0.x
        0xBF90_0007, // s_sendmsg sendmsg(MSG_ORDERED_PS_DONE)
        0xF800_180F,
        0x0404_0404, // exp mrt0 v4, v4, v4, v4 done vm
        0xBF81_0000,
    ];
    let out = lower_ps(
        &direct_m0,
        true,
        vec![],
        ps_in.clone(),
        1 << 1,
        0,
        sig_in.clone(),
        false,
    )
    .unwrap();
    assert!(!out.rovs);

    // 2. Spilled through `v_writelane_b32` / `v_readlane_b32` into `m0`.
    let spilled_m0 = [
        0xD761_0007,
        0x0001_0000, // v_writelane_b32 v7, s0, 0
        0xD760_0001,
        0x0001_0107, // v_readlane_b32 s1, v7, 0
        0xBEFC_0301, // s_mov_b32 m0, s1
        0xC810_0000, // v_interp_p1_f32 v4, v0, attr0.x
        0xC811_0001, // v_interp_p2_f32 v4, v1, attr0.x
        0xF800_180F,
        0x0404_0404,
        0xBF81_0000,
    ];
    lower_ps(&spilled_m0, true, vec![], ps_in, 1 << 1, 0, sig_in, false).unwrap();

    // 3. Guessed PS with POPS (`src_pops_exiting_wave_id` + `MSG_ORDERED_PS_DONE`).
    let guessed_pops = [
        0xBF06_EF09, // s_cmp_eq_u32 s9, src_pops_exiting_wave_id
        0x8502_8281, // s_cselect_b32 s2, 1, 2
        0x7E00_0202, // v_mov_b32_e32 v0, s2
        0xBF90_0007, // s_sendmsg sendmsg(MSG_ORDERED_PS_DONE)
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    let container =
        graphics::container(HwStage::Ps, &bytes_of(&guessed_pops), &[0u8; 0x10]).unwrap();
    translate_with(&container, Options::default()).unwrap();
}

#[test]
fn refusal3_non_interp_hardware_sgpr_remains_refused() {
    // 1. Live-in s0 read by VALU and exported.
    let alu_sgpr = [
        0x7E00_0200, // v_mov_b32_e32 v0, s0
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    let err = lower_ps(&alu_sgpr, true, vec![], vec![], 0, 0, vec![], false)
        .err()
        .expect("expected refusal");
    assert!(err.to_string().contains("a hardware SGPR input"), "{err}");

    // 2. Non-guessed PS with db_shader_control == 0 reading collision id s9.
    let pops_no_ctrl = [
        0x7E00_0209, // v_mov_b32_e32 v0, s9
        0xF800_180F,
        0x0000_0000,
        0xBF81_0000,
    ];
    let err = lower_ps(&pops_no_ctrl, true, vec![], vec![], 0, 0, vec![], false)
        .err()
        .expect("expected refusal");
    assert!(err.to_string().contains("a hardware SGPR input"), "{err}");

    // 3. Live-in s0 moved to m0 and read by both v_interp AND v_movrels_b32.
    let ps_in = vec![PsInput {
        usage: Usage::Generic,
        row: 0,
        mask: 1,
        attr: 0,
        flat: false,
        fp16: false,
        pack2: false,
        row2: 0,
    }];
    let sig_in = vec![sig_el("TEXCOORD", 0, 0, 9, 0, Some(0), 4)];
    let m0_movrels = [
        0xBEFC_0300, // s_mov_b32 m0, s0
        0xC810_0000, // v_interp_p1_f32 v4, v0, attr0.x
        0xC811_0001, // v_interp_p2_f32 v4, v1, attr0.x
        0x7E0A_8700, // v_movrels_b32_e32 v5, v0
        0x0608_0B04, // v_add_f32_e32 v4, v4, v5
        0xF800_180F,
        0x0404_0404,
        0xBF81_0000,
    ];
    let err = lower_ps(&m0_movrels, true, vec![], ps_in, 1 << 1, 0, sig_in, false)
        .err()
        .expect("expected refusal");
    assert!(err.to_string().contains("a hardware SGPR input"), "{err}");
}

#[test]
fn guessed_pixel_masked_prim_mask_anchors_the_pops_input() {
    let code = [
        0x8705_FF06,
        0x0000_FFFF, // s_and_b32 s5, s6, 0xffff
        0xBEFC_0305, // s_mov_b32 m0, s5
        0xC810_0000, // v_interp_p1_f32 v4, v0, attr0.x
        0xC811_0001, // v_interp_p2_f32 v4, v1, attr0.x
        0xBF06_EF07, // s_cmp_eq_u32 s7, src_pops_exiting_wave_id
        0x8512_8081, // s_cselect_b32 s18, 1, 0
        0x7E0A_0212, // v_mov_b32 v5, s18
        0xBF90_0007, // s_sendmsg sendmsg(MSG_ORDERED_PS_DONE)
        0xF800_180F,
        0x0404_0504, // exp mrt0 v4, v5, v4, v4 done vm
        0xBF81_0000,
    ];
    let lower = |code: &[u32]| {
        let container = graphics::container(HwStage::Ps, &bytes_of(code), &[0; 0x10]).unwrap();
        translate_with(&container, Options::default())
    };
    lower(&code).unwrap();

    let mut mixed = code;
    mixed[7] = 0x7E0A_0206; // v_mov_b32 v5, s6: the mask also escapes into the export.
    let err = lower(&mixed).err().unwrap();
    assert!(err.to_string().contains("a hardware SGPR input"), "{err}");

    let mut scc_use = code;
    scc_use.swap(5, 6); // The select now consumes the SCC result of the mask operation.
    let err = lower(&scc_use).err().unwrap();
    assert!(err.to_string().contains("a hardware SGPR input"), "{err}");
}

#[test]
fn guessed_pixel_extracted_prim_mask_anchors_the_pops_input() {
    // Bit-field extraction is another way to prepare M0 for interpolation.
    // The collision ID is s7 because the primitive mask is live-in s6.
    let code = [
        0x9385_FF06,
        0x0010_0000, // s_bfe_u32 s5, s6, 0x00100000 (offset 0, width 16)
        0xBEFC_0305, // s_mov_b32 m0, s5
        0xC810_0000, // v_interp_p1_f32 v4, v0, attr0.x
        0xC811_0001, // v_interp_p2_f32 v4, v1, attr0.x
        0xBF06_EF07, // s_cmp_eq_u32 s7, src_pops_exiting_wave_id
        0x8512_8081, // s_cselect_b32 s18, 1, 0
        0x7E0A_0212, // v_mov_b32 v5, s18
        0xBF90_0007, // s_sendmsg sendmsg(MSG_ORDERED_PS_DONE)
        0xF800_180F,
        0x0404_0504, // exp mrt0 v4, v5, v4, v4 done vm
        0xBF81_0000,
    ];
    for wave32_host in [false, true] {
        let lower = |code: &[u32]| {
            let container = graphics::container(HwStage::Ps, &bytes_of(code), &[0; 0x10]).unwrap();
            translate_with(&container, Options { wave32_host, ..Options::default() })
        };
        lower(&code).unwrap();

        let mut mixed = code;
        mixed[7] = 0x7E0A_0206; // The live-in mask also escapes into an export.
        let err = lower(&mixed).expect_err("exported mask must remain refused");
        assert!(err.to_string().contains("a hardware SGPR input"), "{err}");

        let mut extracted = code;
        extracted[7] = 0x7E0A_0205; // The extracted mask escapes into an export.
        let err = lower(&extracted).expect_err("exported extraction must remain refused");
        assert!(err.to_string().contains("a hardware SGPR input"), "{err}");

        let mut scc_use = code;
        scc_use.swap(5, 6); // The select consumes the extraction's SCC result.
        let err = lower(&scc_use).expect_err("observable SCC must remain refused");
        assert!(err.to_string().contains("a hardware SGPR input"), "{err}");
    }
}

// =========================================================================
// Refusal #4: "one input interpolated two ways"
// =========================================================================

#[test]
fn refusal4_dual_interpolated_generic_input_splits_onto_fresh_row() {
    // Interpolates attr0.x with PerspectiveCenter (v[0:1]), PerspectiveCentroid
    // (v[2:3]), AND flat provoking vertex (v_interp_mov_f32 p0).
    let code = [
        0xBEFC_0300, // s_mov_b32 m0, s0
        0xC810_0000, // v_interp_p1_f32 v4, v0, attr0.x
        0xC811_0001, // v_interp_p2_f32 v4, v1, attr0.x
        0xC814_0002, // v_interp_p1_f32 v5, v2, attr0.x
        0xC815_0003, // v_interp_p2_f32 v5, v3, attr0.x
        0xC81A_0002, // v_interp_mov_f32 v6, p0, attr0.x
        0x0608_0B04, // v_add_f32_e32 v4, v4, v5
        0x0608_0D04, // v_add_f32_e32 v4, v4, v6
        0xF800_180F,
        0x0404_0404, // exp mrt0 v4, v4, v4, v4 done vm
        0xBF81_0000,
    ];
    let ps_in = vec![PsInput {
        usage: Usage::Generic,
        row: 0,
        mask: 1,
        attr: 0,
        flat: false,
        fp16: false,
        pack2: false,
        row2: 0,
    }];
    let sig_in = vec![sig_el("TEXCOORD", 0, 0, 9, 0, Some(0), 4)];
    let out = lower_ps(
        &code,
        true,
        vec![],
        ps_in,
        (1 << 1) | (1 << 2),
        0,
        sig_in,
        false,
    )
    .unwrap();
    let isg1 = parsed_isg1(&out.dxil);
    assert_eq!(isg1.len(), 3, "expected 3 split TEXCOORD elements: {isg1:?}");
    let mut rows: Vec<u32> = isg1.iter().filter_map(|e| e.row).collect();
    rows.sort_unstable();
    rows.dedup();
    assert_eq!(rows.len(), 3, "each mode must live on its own row");
    let mut indices: Vec<u32> = isg1.iter().map(|e| e.index).collect();
    indices.sort_unstable();
    indices.dedup();
    assert_eq!(indices, vec![0, 1, 2], "semantic indices must be unique");

    // Also via guessed graphics (`FSPG`) inferring multiple barycentric pairs.
    let container = graphics::container(HwStage::Ps, &bytes_of(&code), &[0u8; 0x10]).unwrap();
    translate_with(&container, Options::default()).unwrap();
}

#[test]
fn refusal4_non_generic_and_non_provoking_interp_remain_refused() {
    let ps_in = vec![PsInput {
        usage: Usage::Generic,
        row: 0,
        mask: 1,
        attr: 0,
        flat: false,
        fp16: false,
        pack2: false,
        row2: 0,
    }];
    // 1. Non-provoking vertex (p10) stays refused.
    let non_provoking = [
        0xBEFC_0300, // s_mov_b32 m0, s0
        0xC81A_0000, // v_interp_mov_f32 v6, p10, attr0.x
        0xF800_180F,
        0x0606_0606,
        0xBF81_0000,
    ];
    let err = lower_ps(
        &non_provoking,
        true,
        vec![],
        ps_in.clone(),
        1 << 1,
        0,
        vec![sig_el("TEXCOORD", 0, 0, 9, 0, Some(0), 4)],
        false,
    )
    .err()
    .expect("expected refusal");
    assert!(
        err.to_string().contains("non-provoking vertex"),
        "got `{err}`"
    );

    // 2. Non-generic input (kind != 0) interpolated two ways stays refused.
    let dual = [
        0xBEFC_0300,
        0xC810_0000,
        0xC811_0001,
        0xC814_0002,
        0xC815_0003,
        0x0608_0B04,
        0xF800_180F,
        0x0404_0404,
        0xBF81_0000,
    ];
    let err = lower_ps(
        &dual,
        true,
        vec![],
        ps_in,
        (1 << 1) | (1 << 2),
        0,
        vec![sig_el("SV_ClipDistance", 0, 6, 9, 0, Some(0), 4)],
        false,
    )
    .err()
    .expect("expected refusal");
    assert!(
        err.to_string().contains("one input interpolated two ways"),
        "got `{err}`"
    );
}

// =========================================================================
// Refusal #5: "an address in an SGPR that is no root parameter"
// =========================================================================

#[test]
fn refusal5_guessed_vs_inferred_heap_lowers_while_nonguessed_refuses() {
    // s_mov_b32 s3, 0
    // s_load_dwordx4 s[4:7], s[2:3], 0x0
    // s_waitcnt lgkmcnt(0)
    // buffer_load_dword v1, v0, s[4:7], 0 idxen
    // s_waitcnt vmcnt(0)
    // exp pos0 v1, v1, v1, v1 done
    // s_endpgm
    let guessed_heap_vs = [
        0xBE83_0380,
        0xF408_0101,
        0xFA00_0000,
        0xBF8C_C07F,
        0xE030_2000,
        0x8001_0100,
        0xBF8C_3F70,
        0xF800_08CF,
        0x0101_0101,
        0xBF81_0000,
    ];
    let container =
        graphics::container(HwStage::Vs, &bytes_of(&guessed_heap_vs), &[0u8; 0x10]).unwrap();
    translate_with(&container, Options::default()).unwrap();

    // Non-guessed VS (`layout = []`) reading from undeclared s[2:3] MUST still refuse!
    let undeclared_load = [
        0xF400_0101,
        0xFA00_0000, // s_load_dword s4, s[2:3], 0x0
        0xBF8C_C07F,
        0x7E02_0204, // v_mov_b32_e32 v1, s4
        0xF800_08CF,
        0x0101_0101,
        0xBF81_0000,
    ];
    let err = lower_vs(&undeclared_load, true, vec![], false)
        .err()
        .expect("expected refusal");
    assert!(
        err.to_string()
            .contains("an address in an SGPR that is no root parameter"),
        "got `{err}`"
    );
}

#[test]
fn guessed_vertex_rebases_a_header_relative_table_alongside_vertex_fetch() {
    let mut meta = [0u8; 0x30];
    meta[0] = 2;
    meta[12..15].copy_from_slice(&[0x40, 4, 1]);
    meta[0x10] = 0x41;
    meta[0x14..0x16].copy_from_slice(&[2, 3]); // CBV in a table
    meta[0x18] = 8; // descriptor at byte 32 of the table
    meta[0x24..0x27].copy_from_slice(&[4, 0, 3]); // four dwords, relative s0, title slot 3
    let code = [
        0xF408_0200,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[0:1], 0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0302,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 32
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE030_2000,
        0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 idxen
        0xF420_0406,
        0xFA00_0000, // s_buffer_load_dword s16, s[12:15], 0
        0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
        0x7E04_0210, // v_mov_b32 v2, s16
        0xF800_08CF,
        0x0202_0201, // exp pos0 v1, v2, v2, v2 done
        0xBF81_0000,
    ];
    let lower = |code: &[u32], meta: &[u8]| {
        let container = graphics::container(HwStage::Vs, &bytes_of(code), meta).unwrap();
        translate_with(&container, Options::default())
    };
    let shader = lower(&code, &meta).unwrap();
    assert!(!shader.heap);
    assert_eq!(shader.user_data.len(), 1);
    assert_eq!(shader.user_data[0].slot, 3);
    assert_eq!(shader.user_data[0].root, RootType::Table);
    assert_eq!(shader.table_entries, [(3, 2)]);
    assert!(parsed_isg1(&shader.dxil).iter().any(|e| e.name == "VBU1"));

    // Reading a descriptor past the 16-entry table window (entry 20 = byte 640 = 0x280,
    // table_offset_dwords = 160) reports table_entries = [(3, 21)] for host re-mirroring.
    let mut wide_meta = meta;
    wide_meta[0x18] = 160;
    let mut wide_code = code;
    wide_code[4] = 0xFA00_0280; // s_load_dwordx4 s[12:15], s[4:5], 640
    let wide_shader = lower(&wide_code, &wide_meta).unwrap();
    assert_eq!(wide_shader.table_entries, [(3, 21)]);
    assert!(wide_shader.table_dynamic.is_empty(), "all offsets are fixed");

    meta[0x25] = 4; // The original absolute interpretation also supports an explicit s4 record.
    assert_eq!(lower(&code, &meta).unwrap().user_data, shader.user_data);

    let unbound = [
        0xF400_0103,
        0xFA00_0000, // s_load_dword s4, s[6:7], 0
        0x7E02_0204, // v_mov_b32 v1, s4
        0xF800_08CF,
        0x0101_0101, // exp pos0 v1, v1, v1, v1 done
        0xBF81_0000,
    ];
    meta[0x25] = 0;
    let err = lower(&unbound, &meta).err().unwrap();
    assert!(err.to_string().contains("an address in an SGPR that is no root parameter"), "{err}");
}

#[test]
fn guessed_vertex_rebases_a_header_relative_root_view() {
    let mut meta = [0u8; 0x30];
    meta[0] = 2;
    meta[12..15].copy_from_slice(&[0x40, 4, 1]);
    meta[0x10] = 0x41;
    meta[0x14..0x16].copy_from_slice(&[2, 1]); // Root CBV
    meta[0x26] = 3; // Title slot 3
    let code = bytes_of(&[
        0xBE88_0404, // s_mov_b64 s[8:9], s[4:5]
        0xBE8A_03C1, // s_mov_b32 s10, -1
        0xBE8B_03FF,
        0x3001_4FAC, // s_mov_b32 s11, 0x30014fac
        0xF420_0304,
        0xFA00_0000, // s_buffer_load_dword s12, s[8:11], 0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x7E02_020C, // v_mov_b32 v1, s12
        0xF800_08CF,
        0x0101_0101, // exp pos0 v1, v1, v1, v1 done
        0xBF81_0000,
    ]);
    let lower = |meta: &[u8]| {
        let container = graphics::container(HwStage::Vs, &code, meta).unwrap();
        translate_with(&container, Options::default())
    };
    assert_eq!(lower(&meta).unwrap().root_cbvs, [3]);
    meta[13] = 6;
    let err = lower(&meta).err().unwrap();
    assert!(err.to_string().contains("an address in an SGPR that is no root parameter"), "{err}");
}

// =========================================================================
// Refusal #6: "a vertex fetch format the host's vertex layout lacks"
// =========================================================================

/// Encodes `tbuffer_load_format_{x,xy,xyz,xyzw} v[1..], v0, s[4:7], 0 format:<fmt> idxen`.
fn tbuffer_fetch_vs(fmt: u8, count: u32) -> [u32; 10] {
    let w0 = 0xE800_2000 | (u32::from(fmt) << 19) | ((count - 1) << 16);
    let w1 = 0x8001_0100;
    let exp_v = match count {
        1 => 0x0101_0101,
        2 => 0x0202_0201,
        3 => 0x0303_0201,
        _ => 0x0403_0201,
    };
    [
        0xF408_0100,
        0xFA00_0000, // s_load_dwordx4 s[4:7], s[0:1], 0x0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        w0,
        w1,          // tbuffer_load_format_* v[1..], v0, s[4:7], 0 format:<fmt> idxen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xF800_08CF,
        exp_v,       // exp pos0 ... done
        0xBF81_0000, // s_endpgm
        0xBF80_0000, // s_nop 0 (padding)
    ]
}

#[test]
fn refusal6_all_42_vertex_fetch_formats_and_partial_loads_lower() {
    // (GFX10 unified format number, load count, expected semantic name, expected comp, expected cols)
    let cases: &[(u8, u32, &str, u8, u8)] = &[
        // 8-bit 1/2/4 channels
        (1, 1, "VBN1", 9, 1),
        (2, 1, "VBQ1", 9, 1),
        (5, 1, "VBB1", 5, 1),
        (6, 1, "VBC1", 4, 1),
        (14, 2, "VBN2", 9, 2),
        (15, 2, "VBQ2", 9, 2),
        (18, 2, "VBB2", 5, 2),
        (19, 2, "VBC2", 4, 2),
        (56, 4, "VBN4", 9, 4),
        (57, 4, "VBQ4", 9, 4),
        (60, 4, "VBB4", 5, 4),
        (61, 4, "VBC4", 4, 4),
        // 16-bit 1/2/4 channels
        (7, 1, "VBM1", 9, 1),
        (8, 1, "VBS1", 9, 1),
        (11, 1, "VBW1", 5, 1),
        (12, 1, "VBK1", 4, 1),
        (13, 1, "VBH1", 9, 1),
        (23, 2, "VBM2", 9, 2),
        (24, 2, "VBS2", 9, 2),
        (27, 2, "VBW2", 5, 2),
        (28, 2, "VBK2", 4, 2),
        (29, 2, "VBH2", 9, 2),
        (65, 4, "VBM4", 9, 4),
        (66, 4, "VBS4", 9, 4),
        (69, 4, "VBW4", 5, 4),
        (70, 4, "VBK4", 4, 4),
        (71, 4, "VBH4", 9, 4),
        // 32-bit 1/2/3/4 channels
        (20, 1, "VBU1", 5, 1),
        (21, 1, "VBI1", 4, 1),
        (22, 1, "VBF1", 9, 1),
        (62, 2, "VBU2", 5, 2),
        (63, 2, "VBI2", 4, 2),
        (64, 2, "VBF2", 9, 2),
        (72, 3, "VBU3", 5, 3),
        (73, 3, "VBI3", 4, 3),
        (74, 3, "VBF3", 9, 3),
        (75, 4, "VBU4", 5, 4),
        (76, 4, "VBI4", 4, 4),
        (77, 4, "VBF4", 9, 4),
        // Packed formats
        (36, 3, "VBG3", 9, 3), // [11, 11, 10] Float
        (50, 4, "VBP4", 9, 4), // [10, 10, 10, 2] Unorm
        (51, 4, "VBT4", 9, 4), // [10, 10, 10, 2] Snorm
        // Partial channel loads (w.len() > count)
        (71, 3, "VBH4", 9, 4), // xyz of [16,16,16,16] Float
        (50, 3, "VBP4", 9, 4), // xyz of [10,10,10,2] Unorm
        (56, 3, "VBN4", 9, 4), // xyz of [8,8,8,8] Unorm
    ];
    for &(fmt, count, expect_name, expect_comp, expect_cols) in cases {
        let code = tbuffer_fetch_vs(fmt, count);
        for rect_list in [false, true] {
            let out = lower_vs(&code, true, vertex_table_layout(), rect_list)
                .unwrap_or_else(|e| panic!("fmt={fmt} count={count} rect={rect_list}: {e}"));
            let isg1 = parsed_isg1(&out.dxil);
            let el = isg1
                .iter()
                .find(|e| e.name == expect_name)
                .unwrap_or_else(|| {
                    panic!("fmt={fmt} count={count}: missing `{expect_name}` in {isg1:?}")
                });
            assert_eq!(el.comp, expect_comp, "{expect_name}");
            assert_eq!(el.cols, expect_cols, "{expect_name}");
        }
    }
}

#[test]
fn refusal6_unsupported_vertex_fetch_formats_remain_refused() {
    let bad_tbuffer: &[(u8, u32, &str)] = &[
        (58, 4, "8_8_8_8 Uscaled"),
        (59, 4, "8_8_8_8 Sscaled"),
        (44, 4, "2_10_10_10 Unorm (low 2-bit)"),
        (43, 3, "10_11_11 Float (reversed)"),
        (54, 4, "10_10_10_2 Uint"),
        (55, 4, "10_10_10_2 Sint"),
        (29, 4, "xyzw load from 2-channel 16_16 Float (count > w.len())"),
    ];
    for &(fmt, count, label) in bad_tbuffer {
        let code = tbuffer_fetch_vs(fmt, count);
        let err = lower_vs(&code, true, vertex_table_layout(), false)
            .err()
            .unwrap_or_else(|| panic!("{label}: expected refusal"));
        assert!(
            err.to_string()
                .contains("a vertex fetch format the host's vertex layout lacks"),
            "{label}: got `{err}`"
        );
    }

    // Untyped `buffer_load_format_xyzw` (format == None) must remain refused.
    let untyped_format = [
        0xF408_0100,
        0xFA00_0000, // s_load_dwordx4 s[4:7], s[0:1], 0x0
        0xBF8C_C07F,
        0xE00C_2000,
        0x8001_0100, // buffer_load_format_xyzw v[1:4], v0, s[4:7], 0 idxen
        0xBF8C_3F70,
        0xF800_08CF,
        0x0403_0201,
        0xBF81_0000,
    ];
    let err = lower_vs(&untyped_format, true, vertex_table_layout(), false)
        .err()
        .expect("expected refusal");
    assert!(
        err.to_string()
            .contains("a vertex fetch format the host's vertex layout lacks"),
        "untyped buffer_load_format: got `{err}`"
    );
}
