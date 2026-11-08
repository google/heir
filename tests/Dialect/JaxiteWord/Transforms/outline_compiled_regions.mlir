// RUN: heir-opt --jaxiteword-outline-compiled-regions="minimum-operations=2 maximum-operations=10 maximum-regions=2" %s | FileCheck %s --check-prefix=IR
// RUN: heir-opt --jaxiteword-outline-compiled-regions="minimum-operations=2 maximum-operations=10 maximum-regions=2" %s | heir-translate --allow-unregistered-dialect --emit-jaxiteword | FileCheck %s --check-prefix=PY

!Z1032955396097_i64 = !mod_arith.int<1032955396097 : i64>
!Z1095233372161_i64 = !mod_arith.int<1095233372161 : i64>
#encoding = #lwe.inverse_canonical_encoding<scaling_factor = 29>
#key = #lwe.key<>
#ring_pt = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**8>>
!rns_L1 = !rns.rns<!Z1095233372161_i64, !Z1032955396097_i64>
!rns_L0 = !rns.rns<!Z1095233372161_i64>
#ring_ct_L1 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**8>>
#ring_ct_L0 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**8>>
#space_L1 = #lwe.ciphertext_space<ring = #ring_ct_L1, encryption_type = mix>
#space_L0 = #lwe.ciphertext_space<ring = #ring_ct_L0, encryption_type = mix>
#chain_L1 = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64>, current = 1>
#chain_L0 = #lwe.modulus_chain<elements = <1095233372161 : i64>, current = 0>
!ct_L1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_pt, encoding = #encoding>, ciphertext_space = #space_L1, key = #key, modulus_chain = #chain_L1>
!ct_L0 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_pt, encoding = #encoding>, ciphertext_space = #space_L0, key = #key, modulus_chain = #chain_L0>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_pt, encoding = #encoding>>

// IR: func.func @outline_first_region
// IR: [[REGION:%.+]] = jaxiteword.compiled_region
// IR-SAME: regionId = 0 : i64
// IR: jaxiteword.mul_plain
// IR: jaxiteword.rot
// IR: jaxiteword.add
// IR: jaxiteword.add_plain
// IR: jaxiteword.mod_reduce
// IR: jaxiteword.mul_no_relin
// IR: jaxiteword.relin
// IR: jaxiteword.compiled_region_yield

// PY: def _compiled_region_0_raw(
// PY: ._mul_array(
// PY: ._rotate_array(
// PY: ._add_array(
// PY: ._add_plain_array(
// PY: ._rescale_array(
// PY: ._hemul_no_relin_array(
// PY: ._relinearize_array(
// PY: jax.jit(_compiled_region_0_raw, device=_compiled_array_device(
// PY: _prepare_compiled_output_template(
// PY-SAME: _compiled_scale_mul(
// PY-SAME: "_ckks_nsd"
// PY: raise _CompiledPlanPrepared(
// PY: _output_templates[0]._clone_with_payload(
// PY: def outline_first_region__prepare_compiled(ctx, preprocessed_args, input_specs, device=None):
// PY: return _CompiledExecutionPlan(ctx, _execute, _regions, _input_specs,
// PY-SAME: raw_input_path=(0,), raw_output_kind="scalar"
// PY: def outline_first_region__execute_compiled(plan, encrypted_args):
// PY: ._level_reduce_array(
func.func @outline_first_region(
    %ctx: !jaxiteword.crypto_context<>, %ct: !ct_L1, %pt: !pt,
    %bias: !pt, %ek: !jaxiteword.eval_key<>) -> !ct_L0 {
  %unused = func.call @unused_pure_helper() : () -> i32
  %mul = jaxiteword.mul_plain %ctx, %ct, %pt : (!jaxiteword.crypto_context<>, !ct_L1, !pt) -> !ct_L1
  %rot = jaxiteword.rot %ctx, %mul, %ek {index = 1 : i64} : (!jaxiteword.crypto_context<>, !ct_L1, !jaxiteword.eval_key<>) -> !ct_L1
  %add = jaxiteword.add %ctx, %mul, %rot : (!jaxiteword.crypto_context<>, !ct_L1, !ct_L1) -> !ct_L1
  %biased = jaxiteword.add_plain %ctx, %add, %bias : (!jaxiteword.crypto_context<>, !ct_L1, !pt) -> !ct_L1
  %scaled = jaxiteword.mod_reduce %ctx, %biased : (!jaxiteword.crypto_context<>, !ct_L1) -> !ct_L0
  %boundary = jaxiteword.mul_no_relin %ctx, %scaled, %scaled : (!jaxiteword.crypto_context<>, !ct_L0, !ct_L0) -> !ct_L0
  %activated = jaxiteword.relin %ctx, %boundary, %ek : (!jaxiteword.crypto_context<>, !ct_L0, !jaxiteword.eval_key<>) -> !ct_L0
  return %activated : !ct_L0
}

func.func @unused_pure_helper() -> i32 {
  %zero = arith.constant 0 : i32
  return %zero : i32
}

func.func @outline_level_reduce(
    %ctx: !jaxiteword.crypto_context<>, %ct: !ct_L1, %pt: !pt) -> !ct_L0 {
  %mul = jaxiteword.mul_plain %ctx, %ct, %pt : (!jaxiteword.crypto_context<>, !ct_L1, !pt) -> !ct_L1
  %reduced = jaxiteword.level_reduce %ctx, %mul : (!jaxiteword.crypto_context<>, !ct_L1) -> !ct_L0
  return %reduced : !ct_L0
}
