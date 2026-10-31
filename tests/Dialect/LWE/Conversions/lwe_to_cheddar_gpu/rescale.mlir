// RUN: heir-opt --lwe-to-cheddar-gpu %s | FileCheck %s

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
#ring_f64_1_x65536 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**65536>>

!rns_L1 = !rns.rns<!mod_arith.int<1125899908022273 : i64>, !mod_arith.int<1099515691009 : i64>>
!rns_L2 = !rns.rns<!mod_arith.int<1125899908022273 : i64>, !mod_arith.int<1099515691009 : i64>, !mod_arith.int<1099523555329 : i64>>

#ring_rns_L1 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**65536>>
#ring_rns_L2 = #polynomial.ring<coefficientType = !rns_L2, polynomialModulus = <1 + x**65536>>

#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1, encryption_type = mix>
#ciphertext_space_L2 = #lwe.ciphertext_space<ring = #ring_rns_L2, encryption_type = mix>

#chain_L1 = #lwe.modulus_chain<elements = <1125899908022273 : i64, 1099515691009 : i64>, current = 1>
#chain_L1_lr = #lwe.modulus_chain<elements = <1125899908022273 : i64, 1099515691009 : i64, 1099523555329 : i64>, current = 1>
#chain_L2 = #lwe.modulus_chain<elements = <1125899908022273 : i64, 1099515691009 : i64, 1099523555329 : i64>, current = 2>

!ct_L1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #chain_L1>
!ct_L1_lr = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #chain_L1_lr>
!ct_L2 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L2, key = #key, modulus_chain = #chain_L2>

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 16, Q = [1125899908022273, 1099515691009, 1099523555329], P = [72057594038321153], logDefaultScale = 40>} {
  // CHECK: func.func @test_rescale
  // CHECK-SAME: (%[[OUT_BX:[^,]+]]: memref<2x65536xi64>, %[[OUT_AX:[^,]+]]: memref<2x65536xi64>, %[[IN_BX:[^,]+]]: memref<3x65536xi64>, %[[IN_AX:[^)]+]]: memref<3x65536xi64>)
  // CHECK: cheddar_gpu.rescale %[[OUT_BX]], %[[IN_BX]] <level = 2,
  // CHECK: cheddar_gpu.rescale %[[OUT_AX]], %[[IN_AX]] <level = 2,
  // CHECK-NOT: memref.copy
  func.func @test_rescale(%a: !ct_L2) -> !ct_L1 {
    %0 = ckks.rescale %a <to_ring = #ring_rns_L1> : !ct_L2 -> !ct_L1
    return %0 : !ct_L1
  }

  // Per Context.cpp LevelDown:
  // level_reduce ≡ Context::LevelDown = n * [mul_const(EncodeConstant(1.0, L)); rescale]
  // CHECK: func.func @test_level_reduce
  // CHECK: cheddar_gpu.cp_accum_2
  // CHECK: cheddar_gpu.rescale {{.*}} <level = 2,
  // CHECK: cheddar_gpu.rescale {{.*}} <level = 2,
  func.func @test_level_reduce(%a: !ct_L2) -> !ct_L1_lr {
    %0 = ckks.level_reduce %a <levelToDrop = 1> : !ct_L2 -> !ct_L1_lr
    return %0 : !ct_L1_lr
  }
}
