// RUN: heir-opt --lwe-to-cheddar-gpu %s | FileCheck %s

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
#ring_f64_1_x65536 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**65536>>

!rns_L2 = !rns.rns<!mod_arith.int<1125899908022273 : i64>, !mod_arith.int<1099515691009 : i64>, !mod_arith.int<1099523555329 : i64>>
#ring_rns_L2 = #polynomial.ring<coefficientType = !rns_L2, polynomialModulus = <1 + x**65536>>
#ciphertext_space_L2 = #lwe.ciphertext_space<ring = #ring_rns_L2, encryption_type = mix>
#chain_L2 = #lwe.modulus_chain<elements = <1125899908022273 : i64, 1099515691009 : i64, 1099523555329 : i64>, current = 2>

!ct_L2 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L2, key = #key, modulus_chain = #chain_L2>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding>>

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 16, Q = [1125899908022273, 1099515691009, 1099523555329], P = [72057594038321153], logDefaultScale = 40>} {
  // CHECK: func.func @test_hoist_encode
  // CHECK-SAME: (%[[OUT_BX:[^,]+]]: memref<3x65536xi64>, %[[OUT_AX:[^,]+]]: memref<3x65536xi64>, %[[IN_BX:[^,]+]]: memref<3x65536xi64>, %[[IN_AX:[^,]+]]: memref<3x65536xi64>, %[[PT:[^)]+]]: memref<3x65536xi64>)
  // CHECK-NOT: lwe.rlwe_encode
  // CHECK: cheddar_gpu.cp_accum_2
  func.func @test_hoist_encode(%a: !ct_L2) -> !ct_L2 {
    %cst = arith.constant dense<[1.0, 2.0]> : tensor<2xf64>
    %pt = lwe.rlwe_encode %cst <encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x65536, level = 2, scale = 0> : tensor<2xf64> -> !pt
    %0 = ckks.mul_plain %a, %pt : (!ct_L2, !pt) -> !ct_L2
    return %0 : !ct_L2
  }
}
