// RUN: heir-opt --cggi-to-openfhe %s | FileCheck %s

#preserve_overflow = #lwe.preserve_overflow<>
#key = #lwe.key<slot_index = 0>
#poly = #polynomial.int_polynomial<1 + x**1024>
#pspace = #lwe.plaintext_space<ring = #polynomial.ring<coefficientType = i3, polynomialModulus = #poly>, encoding = #lwe.constant_coefficient_encoding<scaling_factor = 268435456>>
#cspace = #lwe.ciphertext_space<ring = #polynomial.ring<coefficientType = i32, polynomialModulus = #poly>, encryption_type = msb, size = 2>
!ct_ty = !lwe.lwe_ciphertext<application_data = <message_type = i3, overflow = #preserve_overflow>, plaintext_space = #pspace, ciphertext_space = #cspace, key = #key>

// CHECK: func.func @test_lut_lincomb
// CHECK: openfhe.get_lwe_scheme
// CHECK: arith.constant 1 : i64
// CHECK: openfhe.lwe_mul_const
// CHECK: arith.constant 2 : i64
// CHECK: openfhe.lwe_mul_const
// CHECK: arith.constant 3 : i64
// CHECK: openfhe.lwe_mul_const
// CHECK: arith.constant 2 : i64
// CHECK: openfhe.lwe_mul_const
// CHECK: openfhe.lwe_add
// CHECK: openfhe.lwe_add
// CHECK: openfhe.lwe_add
// CHECK: openfhe.make_lut {{.*}} {values = array<i32: 2, 6>}
// CHECK: openfhe.eval_func
// CHECK: return
func.func @test_lut_lincomb(%arg0: !ct_ty, %arg1: !ct_ty, %arg2: !ct_ty, %arg3: !ct_ty) -> !ct_ty {
  %0 = cggi.lut_lincomb %arg0, %arg1, %arg2, %arg3 {coefficients = array<i32: 1, 2, 3, 2>, lookup_table = 68 : index} : !ct_ty
  return %0 : !ct_ty
}
