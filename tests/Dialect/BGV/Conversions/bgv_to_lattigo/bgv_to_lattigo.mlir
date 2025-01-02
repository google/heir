// RUN: heir-opt --mlir-print-local-scope --bgv-to-lattigo %s | FileCheck %s


!Z1032955396097_i64_ = !mod_arith.int<1032955396097 : i64>
!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>

!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
!rns_L1_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_>
#ring_Z65537_i64_1_x1024_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**1024>>
#ring_rns_L0_1_x1024_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**1024>>
#ring_rns_L1_1_x1024_ = #polynomial.ring<coefficientType = !rns_L1_, polynomialModulus = <1 + x**1024>>

#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>

#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>
#modulus_chain_L5_C1_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 1>

#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x1024_, encoding = #full_crt_packing_encoding>

#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024_, encryption_type = lsb>
#ciphertext_space_L1_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb>
#ciphertext_space_L1_D3_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb, size = 3>

!ct = !lwe.new_lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L5_C1_>
!ct1 = !lwe.new_lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_D3_, key = #key, modulus_chain = #modulus_chain_L5_C1_>
!ct2 = !lwe.new_lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

// CHECK: module
module {
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
    %rescale = bgv.modulus_switch %relin {to_ring = #ring_rns_L0_1_x1024_} : !ct -> !ct2
    // CHECK: %[[rot:.*]] = lattigo.bgv.rotate_columns [[C]], %[[rescale]] {offset = 1 : i64} : ([[S]], [[T]]) -> [[T]]
    %rot = bgv.rotate %rescale { offset = 1 } : !ct2
    return
  }
}