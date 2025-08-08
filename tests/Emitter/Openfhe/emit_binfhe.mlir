// RUN: heir-translate %s --emit-openfhe-bin | FileCheck %s

#preserve_overflow = #lwe.preserve_overflow<>
#key = #lwe.key<slot_index = 0>
#poly = #polynomial.int_polynomial<1 + x**1024>
#pspace = #lwe.plaintext_space<
  ring = #polynomial.ring<coefficientType = i3, polynomialModulus = #poly>,
  encoding = #lwe.constant_coefficient_encoding<scaling_factor = 268435456>
>
#cspace = #lwe.ciphertext_space<
  ring = #polynomial.ring<coefficientType = i32, polynomialModulus = #poly>,
  encryption_type = msb,
  size = 2
>
!ct = !lwe.lwe_ciphertext<
  application_data = <message_type = i3, overflow = #preserve_overflow>,
  plaintext_space = #pspace,
  ciphertext_space = #cspace,
  key = #key
>
!ctx = !openfhe.binfhe_context
!scheme = !openfhe.lwe_scheme
!lut = !openfhe.lookup_table

module {

  // CHECK: CiphertextT test_basic_lut(
  // CHECK: GetLWEScheme()
  // CHECK: GenerateLUTviaFunction
  // CHECK: EvalFunc
  func.func @test_basic_lut(%ctx: !ctx, %arg0: !ct) -> !ct {
    %scheme = openfhe.get_lwe_scheme %ctx : (!ctx) -> !scheme
    %lut = openfhe.make_lut %ctx {values = array<i32: 0, 1, 2, 3, 4, 5, 6, 7>} : (!ctx) -> !lut
    %result = openfhe.eval_func %ctx, %lut, %arg0 : (!ctx, !lut, !ct) -> !ct
    return %result : !ct
  }

  // CHECK: CiphertextT test_lwe_ops(
  // CHECK: GetLWEScheme()
  // CHECK: EvalMultConstEq
  // CHECK: EvalAddEq
  func.func @test_lwe_ops(%ctx: !ctx, %arg0: !ct, %arg1: !ct) -> !ct {
    %scheme = openfhe.get_lwe_scheme %ctx : (!ctx) -> !scheme
    %c3 = arith.constant 3 : i64
    %mul_result = openfhe.lwe_mul_const %scheme, %arg0, %c3 : (!scheme, !ct, i64) -> !ct
    %add_result = openfhe.lwe_add %scheme, %mul_result, %arg1 : (!scheme, !ct, !ct) -> !ct
    return %add_result : !ct
  }
}
