# AMD GFX906 (MI50/Vega 20) Undocumented ISA Instructions
## Comprehensive Hardware Probe Results — March 2026 - David Sanftenberg

### Methodology
- **Target**: AMD Instinct MI50 (GFX906, sramecc+:xnack-, 32GB, 60 CUs)
- **Toolchain**: ROCm hipcc + LLVM assembler
- **Approach**: Binary `.byte` injection into HIP kernels, per-process isolation for ILLEGAL handling
- **Validation**: Multi-vector characterization (5+ test vectors per opcode), single-bit transfer functions
- **Encoding**: VOP3 format — byte3=0xD1 for all vector ALU ops

### Coverage Summary

| Encoding Space | Probed | VALID+Undocumented | ILLEGAL | Documented |
|---|---|---|---|---|
| VOP3 D1 VOP2-range (0x00-0x3F) | 0x17-0x3F | **16** | 11 | ~35 |
| VOP3 D1 Dead Zone (0x8B-0xBF) | 0x8B-0xBF | **6** | ~47 | 0 |
| VOP3P (0x80-0xBF) | 0x80-0xBF | **0** | ~4 | ~44 |
| **TOTAL** | | **22** | | |

---

## Category 1: GFX900 Legacy INT16/FP16 VOP2 Instructions (16 ops)

These are GFX900 (Vega 10) instructions that were officially deprecated in GFX906's ISA reference
but remain **fully functional in silicon**. They were replaced by VOP3-only equivalents on GFX906,
but the VOP2→VOP3 promotion decoder was never removed from the hardware.

### Verified via 5-vector characterization

| D1 Opcode | VOP2 Opcode | Instruction | Operation | Confidence |
|---|---|---|---|---|
| **0x23** | 0x23 | `v_mac_f16` | `dst.f16 += src0.f16 × src1.f16` | ✅ Confirmed |
| **0x26** | 0x26 | `v_add_u16` | `dst.u16 = src0.u16 + src1.u16` | ✅ Confirmed |
| **0x27** | 0x27 | `v_sub_u16` | `dst.u16 = src0.u16 − src1.u16` | ✅ Confirmed |
| **0x28** | 0x28 | `v_subrev_u16` | `dst.u16 = src1.u16 − src0.u16` | ✅ Confirmed |
| **0x29** | 0x29 | `v_mul_lo_u16` | `dst.u16 = (src0.u16 × src1.u16) & 0xFFFF` | ✅ Confirmed |
| **0x2A** | 0x2A | `v_lshlrev_b16` | `dst.u16 = src1.u16 << (src0 & 0xF)` | ✅ Confirmed |
| **0x2B** | 0x2B | `v_lshrrev_b16` | `dst.u16 = src1.u16 >> (src0 & 0xF)` | ✅ Confirmed |
| **0x2C** | 0x2C | `v_ashrrev_i16` | `dst.i16 = (i16)src1 >> (src0 & 0xF)` | ✅ Confirmed |
| **0x2F** | 0x2F | `v_max_u16` | `dst.u16 = max(src0.u16, src1.u16)` | ✅ Confirmed |
| **0x30** | 0x30 | `v_max_i16` | `dst.i16 = max((i16)src0, (i16)src1)` | ✅ Confirmed |
| **0x31** | 0x31 | `v_min_u16` | `dst.u16 = min(src0.u16, src1.u16)` | ✅ Confirmed |
| **0x32** | 0x32 | `v_min_i16` | `dst.i16 = min((i16)src0, (i16)src1)` | ✅ Confirmed |

### GFX900 Legacy 32-bit No-Carry Arithmetic (VOP2→VOP3)

These were moved from VOP2 to VOP3-only on GFX9 (the VOP2 slots reused for carry variants).
The old VOP2 decoders still work.

| D1 Opcode | VOP2 Opcode | Instruction | Operation | Confidence |
|---|---|---|---|---|
| **0x34** | 0x34 | `v_add_u32` (no carry) | `dst = src0 + src1` (no VCC write) | ✅ Confirmed |
| **0x35** | 0x35 | `v_sub_u32` (no carry) | `dst = src0 − src1` (no VCC write) | ✅ Confirmed |
| **0x36** | 0x36 | `v_subrev_u32` (no carry) | `dst = src1 − src0` (no VCC write) | ✅ Confirmed |

### VOP2 Legacy FP16 (special encoding)

| D1 Opcode | Instruction | Note |
|---|---|---|
| **0x24** | — | ILLEGAL (v_madmk_f16 requires literal constant, can't promote to VOP3) |
| **0x25** | — | ILLEGAL (v_madak_f16 same issue) |

---

## Category 2: Dead Zone Instructions (0x8B-0x90) — 6 ops

These are in the **gap between VOP1 promotions (ending at 0x8A) and VOP3-only instructions (starting at 0xC0)**.
They are NOT documented for ANY GFX9 variant. These appear to be silicon-internal helper instructions.

### Fully Characterized

| D1 Opcode | VOP1 Opcode | Instruction (assigned) | Operation | Confidence |
|---|---|---|---|---|
| **0x8F** | 0x4F | `v_cvt_pk_u8_i16` | Packs two signed i16 → two u8 with clamp [0,255] | ✅ Confirmed (18 test vectors) |

**v_cvt_pk_u8_i16 detailed behavior:**
```
src0 treated as two packed signed int16: {hi_i16, lo_i16}
result_lo8 = clamp(lo_i16, 0, 255)
result_hi8 = clamp(hi_i16, 0, 255)
dst = (result_hi8 << 8) | result_lo8   // zero-extended to 32 bits
```
- Negative i16 values → 0 (not wrapped)
- i16 > 255 → 255 (saturated)
- 0 ≤ i16 ≤ 255 → passthrough

**Potential use**: Quantized neural network inference — clipping INT16 accumulation results to UINT8 output range.

### Partially Characterized — FP16 Exponent Scaling

| D1 Opcode | VOP1 Opcode | Instruction (assigned) | Operation | Confidence |
|---|---|---|---|---|
| **0x8D** | 0x4D | `v_exp2_exp_i16_f16` | dst.i16 = sat_i16(2^fp16_biased_exp(src0.lo16)) | ⚠️ Partial |
| **0x8E** | 0x4E | `v_exp2_exp_u16_f16` | dst.u16 = sat_u16(2^(fp16_biased_exp(src0.lo16)+1)) | ⚠️ Partial |

**Behavior**: VOP1-type (src0 only, src1/src2 ignored). Reads lo16(src0) as FP16, extracts the biased exponent field, returns 2 raised to that exponent.

**D1:0x8D single-bit transfer function:**
```
Input bit 9  (FP16 subnormal) → 1      = 2^0
Input bit 10 (FP16 exp=1)     → 2      = 2^1
Input bit 11 (FP16 exp=2)     → 4      = 2^2
Input bit 12 (FP16 exp=4)     → 16     = 2^4
Input bit 13 (FP16 exp=8)     → 256    = 2^8
Input bit 14 (FP16 exp=16)    → 0x7FFF = INT16_MAX (saturated)
```

**D1:0x8E single-bit transfer function:**
```
Input bit 8  (FP16 subnormal) → 1
Input bit 9  (FP16 subnormal) → 2
Input bit 10 (FP16 exp=1)     → 4      = 2^(1+1)
Input bit 11 (FP16 exp=2)     → 8      = 2^(2+1)
Input bit 12 (FP16 exp=4)     → 32     = 2^(4+1)
Input bit 13 (FP16 exp=8)     → 512    = 2^(8+1)
Input bit 14 (FP16 exp=16)    → 0xFFFF = UINT16_MAX (saturated)
```

**Likely purpose**: Internal helpers for FP16 fast reciprocal/log2/exp2 approximation.

### Unconditional Zero-Write (Non-functional)

| D1 Opcode | VOP1 Opcode | Instruction | Notes |
|---|---|---|---|
| **0x8B** | 0x4B | `v_zero_b32` (?) | Always writes 0 to vdst, ignores all sources |
| **0x8C** | 0x4C | `v_zero_b32` (?) | Same — always writes 0 |
| **0x90** | 0x50 | `v_zero_b32` (?) | Same — always writes 0 |

Tested with 21 different input patterns each. All return 0. The canary test confirms these DO write to vdst (not NOPs).
Possibly vestigial decode slots from canceled instructions, or they read from hardware state that is always 0 in compute mode.

---

## Category 3: VOP3P — No Undocumented Instructions Found

An earlier probing session (prior to this document) reported 7 VOP3P opcodes (0xA0-0xA2,
0xA6-0xA7, 0xA9, 0xAB) as "undocumented." **This was incorrect.** LLVM's assembler for
`-mcpu=gfx906` recognizes all 7 with correct mnemonics:

| VOP3P Opcode | LLVM Mnemonic | Earlier (Wrong) Claim |
|---|---|---|
| 0xA0 | `v_fma_mix_f32` | "v_dot1_f32_f16" |
| 0xA1 | `v_fma_mixlo_f16` | "v_dot1_f16_f16_lo" |
| 0xA2 | `v_fma_mixhi_f16` | "v_dot1_f16_f16_hi" |
| 0xA6 | `v_dot2_i32_i16` | Same name, but called undocumented |
| 0xA7 | `v_dot2_u32_u16` | Same name, but called undocumented |
| 0xA9 | `v_dot4_u32_u8` | Same name, but called undocumented |
| 0xAB | `v_dot8_u32_u4` | Same name, but called undocumented |

The earlier session's LLVM sweep used a disassembly approach that didn't enable the right
subtarget features (`dot2-insts`, `dot7-insts`). The assembler (`llvm-mc -mcpu=gfx906`)
correctly encodes all 7 instructions.

**The VOP3P opcode space contains zero undocumented instructions on GFX906.**

---

## Category 4: Complete Opcode Space Coverage

### VOP3 D1 Full Map (byte3=0xD1)

```
0x00-0x16: Documented VOP2 promotions (v_add_f32, v_mul_f32, v_and_b32, etc.)
0x17-0x18: ILLEGAL
0x19-0x22: Documented VOP2 promotions (v_add_co_u32, v_sub_f16, v_mul_f16, etc.)
0x23:      ★ UNDOCUMENTED  v_mac_f16 (GFX900 legacy)
0x24-0x25: ILLEGAL         (v_madmk_f16/v_madak_f16 can't promote to VOP3)
0x26-0x2C: ★ UNDOCUMENTED  v_add_u16 through v_ashrrev_i16 (GFX900 legacy INT16)
0x2D-0x2E: Documented      (v_max_f16, v_min_f16)
0x2F-0x32: ★ UNDOCUMENTED  v_max_u16/i16, v_min_u16/i16 (GFX900 legacy)
0x33:      Documented      (v_ldexp_f16)
0x34-0x36: ★ UNDOCUMENTED  v_add/sub/subrev_u32 no-carry (GFX900 legacy)
0x37-0x3A: ILLEGAL
0x3B:      Documented      (v_fmac_f32)
0x3C:      ILLEGAL
0x3D:      Documented      (v_xnor_b32)
0x3E-0x3F: ILLEGAL
0x40:      Documented      (v_nop VOP3 form)
0x41-0x8A: Documented      VOP1 promotions (v_mov, v_cvt_*, v_rcp, v_sqrt, etc.)
0x8B-0x8C: ★ UNDOCUMENTED  v_zero (always writes 0)
0x8D:      ★ UNDOCUMENTED  v_exp2_exp_i16_f16 (FP16 exponent → 2^exp, sat i16)
0x8E:      ★ UNDOCUMENTED  v_exp2_exp_u16_f16 (FP16 exponent → 2^(exp+1), sat u16)
0x8F:      ★ UNDOCUMENTED  v_cvt_pk_u8_i16 (packed i16→u8 with clamp)
0x90:      ★ UNDOCUMENTED  v_zero (always writes 0)
0x91-0xBF: ILLEGAL         (dead zone — 47 opcodes confirmed ILLEGAL)
0xC0-0xFF: Documented      VOP3-only (v_mad, v_fma, v_bfe, v_perm, etc.)
```

### VOP3P Full Map (byte3=0xD3)

```
0x00-0x7F: ALL ILLEGAL
0x80-0x9F: Documented      (v_pk_* packed 16-bit ops)
0xA0:      Documented      v_fma_mix_f32
0xA1:      Documented      v_fma_mixlo_f16
0xA2:      Documented      v_fma_mixhi_f16
0xA3:      Documented      v_dot2_f32_f16
0xA4-0xA5: ILLEGAL         (disassembler false-decodes as VOP2_e32)
0xA6:      Documented      v_dot2_i32_i16
0xA7:      Documented      v_dot2_u32_u16
0xA8:      Documented      v_dot4_i32_i8
0xA9:      Documented      v_dot4_u32_u8
0xAA:      Documented      v_dot8_i32_i4
0xAB:      Documented      v_dot8_u32_u4
0xAC-0xBF: ILLEGAL
```

---

## Practical Assessment for Llaminar

### Potentially Useful for GEMM/Decode Kernels

| Instruction | Use Case | Impact |
|---|---|---|
| `v_cvt_pk_u8_i16` (0x8F) | Pack INT16 accumulators to UINT8 output | **Medium** — saves extra clamp+pack instructions in quantized output path |
| `v_mul_lo_u16` (0x29) | 16-bit multiply for narrow compute | **Low** — already have 32-bit multiply |
| `v_add/sub_u32` no-carry (0x34-0x36) | Avoids VCC clobber | **Low** — minor scheduling benefit |

### Not Useful

| Category | Reason |
|---|---|
| INT16 arithmetic (0x26-0x32) | Nibble decode bottleneck is 4-bit, not 16-bit |
| FP16 legacy (0x23) | Already have documented v_fma_f16 |
| Dead zone zero-writes (0x8B-0x8C, 0x90) | No computational value |
| FP16 exponent helpers (0x8D-0x8E) | Too specialized, not useful for quantization |
| VOP3P dot products (0xA6-0xAB) | These are actually documented by LLVM; not undocumented after all |

### Key Conclusion

**No undocumented "vectorized nibble ops"** exist in the GFX906 ISA. The nibble decode bottleneck
(AND+SHIFT+SUB, 22 ALU ops per 32 nibbles) cannot be solved by hidden instructions. The most
interesting find is `v_cvt_pk_u8_i16` (0x8F) which could marginally help with quantized output
packing, but the core Q4_0/IQ4_NL decode path remains bound by the documented instruction set.

---

## Encoding Reference

All instructions use VOP3a encoding:
```
First dword:  [31:26]=110100, [25]=clamp, [24:16]=opcode, [15:8]=modifiers, [7:0]=VDST
Second dword: [8:0]=src0, [17:9]=src1, [26:18]=src2

byte3=0xD1 → opcode bit[8]=1 → VOP3 D1 space (all vector ALU)
byte3=0xD3 → VOP3P space (packed operations)

Probe template for D1:
.byte 0x03, 0x00, OPCODE, 0xd1, 0x01, 0x05, 0x0e, 0x04
     (vdst=v3)  (opc)   (D1)  (src0=v1)(src1=v2)(src2=v3)
```
