// RUN: heir-opt --mlir-print-local-scope --bgv-to-lattigo %s | FileCheck %s

#encoding = #lwe.polynomial_evaluation_encoding<cleartext_start=30, cleartext_bitwidth=3>

#my_poly = #polynomial.int_polynomial<1 + x**1024>
// cmod is 64153 * 2521
#ring1 = #polynomial.ring<coefficientType=!mod_arith.int<161729713:i32>, polynomialModulus=#my_poly>
#ring2 = #polynomial.ring<coefficientType=!mod_arith.int<64153:i32>, polynomialModulus=#my_poly>

#params1 = #lwe.rlwe_params<dimension=2, ring=#ring1>
#params2 = #lwe.rlwe_params<dimension=3, ring=#ring1>
#params3 = #lwe.rlwe_params<dimension=2, ring=#ring2>

!ct = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params1, underlying_type=i3>
!ct1 = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params2, underlying_type=i3>
!ct2 = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params3, underlying_type=i3>

// CHECK: module
module {
  // CHECK-LABEL: @test_fn
  // CHECK-SAME: ([[X:%.+]]: [[T:!lattigo.rlwe.ciphertext]]) -> [[T]]
  func.func @test_fn(%x : !ct) -> !ct {
    // CHECK: return [[X]] : [[T]]
    return %x : !ct
  }

  // CHECK-LABEL: @test_ops
  // CHECK-SAME: ([[C:%.+]]: [[S:.*evaluator]], [[X:%.+]]: [[T:!lattigo.rlwe.ciphertext]], [[Y:%.+]]: [[T]])
  func.func @test_ops(%x : !ct, %y : !ct) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.add [[C]], %[[x:.*]], %[[y:.*]]: ([[S]], [[T]], [[T]]) -> [[T]]
    %add = bgv.add %x, %y  : !ct
    // CHECK: %[[mul:.*]] = lattigo.bgv.mul [[C]], %[[x]], %[[y]]: ([[S]], [[T]], [[T]]) -> [[T]]
    %mul = bgv.mul %x, %y  : (!ct, !ct) -> !ct1
    // CHECK: %[[relin:.*]] = lattigo.bgv.relinearize [[C]], %[[mul]] : ([[S]], [[T]]) -> [[T]]
    %relin = bgv.relinearize %mul  {
      from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>
    }: !ct1 -> !ct
    // CHECK: %[[rescale:.*]] = lattigo.bgv.rescale [[C]], %[[relin]] : ([[S]], [[T]]) -> [[T]]
    %rescale = bgv.modulus_switch %relin {to_ring = #ring2} : !ct -> !ct2
    // CHECK: %[[rot:.*]] = lattigo.bgv.rotate_columns [[C]], %[[rescale]] {offset = 1 : i64} : ([[S]], [[T]]) -> [[T]]
    %rot = bgv.rotate %rescale { offset = 1 } : !ct2
    return
  }
}
