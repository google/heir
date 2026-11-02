// RUN: heir-opt --lwe-to-cheddar-gpu %s | FileCheck %s

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
#ring_f64_1_x65536 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**65536>>

!rns_L2 = !rns.rns<!mod_arith.int<1125899908022273 : i64>, !mod_arith.int<1099515691009 : i64>, !mod_arith.int<1099523555329 : i64>>
#ring_rns_L2 = #polynomial.ring<coefficientType = !rns_L2, polynomialModulus = <1 + x**65536>>
#ciphertext_space_L2 = #lwe.ciphertext_space<ring = #ring_rns_L2, encryption_type = mix>
#ciphertext_space_L2_D3 = #lwe.ciphertext_space<ring = #ring_rns_L2, encryption_type = mix, size = 3>
#chain_L2 = #lwe.modulus_chain<elements = <1125899908022273 : i64, 1099515691009 : i64, 1099523555329 : i64>, current = 2>

!ct_L2 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L2, key = #key, modulus_chain = #chain_L2>
!ct_L2_D3 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L2_D3, key = #key, modulus_chain = #chain_L2>
!pt_L2 = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding>>

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 16, Q = [1125899908022273, 1099515691009, 1099523555329], P = [72057594038321153], logDefaultScale = 40>} {
  // CHECK: func.func @test_add
  // CHECK-SAME: (%[[OUT_BX:[^,]+]]: memref<3x65536xi64>, %[[OUT_AX:[^,]+]]: memref<3x65536xi64>, %[[A_BX:[^,]+]]: memref<3x65536xi64>, %[[A_AX:[^,]+]]: memref<3x65536xi64>, %[[B_BX:[^,]+]]: memref<3x65536xi64>, %[[B_AX:[^)]+]]: memref<3x65536xi64>)
  // CHECK: cheddar_gpu.add_2 %[[OUT_BX]], %[[OUT_AX]], %[[A_BX]], %[[A_AX]], %[[B_BX]], %[[B_AX]]
  // CHECK-NOT: cheddar_gpu.copy
  func.func @test_add(%a: !ct_L2, %b: !ct_L2) -> !ct_L2 {
    %0 = ckks.add %a, %b : (!ct_L2, !ct_L2) -> !ct_L2
    return %0 : !ct_L2
  }

  // CHECK: func.func @test_sub(
  // CHECK: cheddar_gpu.sub_2
  func.func @test_sub(%a: !ct_L2, %b: !ct_L2) -> !ct_L2 {
    %0 = ckks.sub %a, %b : (!ct_L2, !ct_L2) -> !ct_L2
    return %0 : !ct_L2
  }

  // CHECK: func.func @test_negate(
  // CHECK: cheddar_gpu.neg_2
  func.func @test_negate(%a: !ct_L2) -> !ct_L2 {
    %0 = ckks.negate %a : !ct_L2
    return %0 : !ct_L2
  }

  // CHECK: func.func @test_add_plain(
  // CHECK: cheddar_gpu.add
  // CHECK: cheddar_gpu.copy
  func.func @test_add_plain(%a: !ct_L2, %pt: !pt_L2) -> !ct_L2 {
    %0 = ckks.add_plain %a, %pt : (!ct_L2, !pt_L2) -> !ct_L2
    return %0 : !ct_L2
  }

  // CHECK: func.func @test_sub_plain_ct_pt(
  // CHECK: cheddar_gpu.sub
  // CHECK: cheddar_gpu.copy
  func.func @test_sub_plain_ct_pt(%a: !ct_L2, %pt: !pt_L2) -> !ct_L2 {
    %0 = ckks.sub_plain %a, %pt : (!ct_L2, !pt_L2) -> !ct_L2
    return %0 : !ct_L2
  }

  // CHECK: func.func @test_sub_plain_pt_ct(
  // CHECK: cheddar_gpu.sub
  // CHECK: cheddar_gpu.neg
  func.func @test_sub_plain_pt_ct(%pt: !pt_L2, %a: !ct_L2) -> !ct_L2 {
    %0 = ckks.sub_plain %pt, %a : (!pt_L2, !ct_L2) -> !ct_L2
    return %0 : !ct_L2
  }

  // CHECK: func.func @test_mul_plain(
  // CHECK: cheddar_gpu.cp_accum_2
  func.func @test_mul_plain(%a: !ct_L2, %pt: !pt_L2) -> !ct_L2 {
    %0 = ckks.mul_plain %a, %pt : (!ct_L2, !pt_L2) -> !ct_L2
    return %0 : !ct_L2
  }

  // CHECK: func.func @test_mul(
  // CHECK: cheddar_gpu.tensor
  func.func @test_mul(%a: !ct_L2, %b: !ct_L2) -> !ct_L2_D3 {
    %0 = ckks.mul %a, %b : (!ct_L2, !ct_L2) -> !ct_L2_D3
    return %0 : !ct_L2_D3
  }
}
