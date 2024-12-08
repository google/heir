// RUN: heir-opt --mlir-print-local-scope --bgv-to-lattigo %s | FileCheck %s

#encoding = #lwe.polynomial_evaluation_encoding<cleartext_start=30, cleartext_bitwidth=3>

#my_poly = #polynomial.int_polynomial<1 + x**1024>
// cmod is 64153 * 2521
#ring1 = #polynomial.ring<coefficientType=!mod_arith.int<161729713:i32>, polynomialModulus=#my_poly>

#params1 = #lwe.rlwe_params<dimension=2, ring=#ring1>

!ct = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params1, underlying_type=i3>

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
    return
  }
}
