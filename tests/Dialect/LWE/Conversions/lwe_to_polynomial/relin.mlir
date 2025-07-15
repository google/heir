// RUN: heir-opt --lwe-to-polynomial %s | FileCheck %s

!Z1032955396097_i64_ = !mod_arith.int<1032955396097 : i64>
!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>

!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
!rns_L1_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_>

#ring_Z65537_i64_1_x1024_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**1024>>
#ring_rns_L0_1_x1024_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**1024>>
#ring_rns_L1_1_x1024_ = #polynomial.ring<coefficientType = !rns_L1_, polynomialModulus = <1 + x**1024>>

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>

#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>
#modulus_chain_L5_C1_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 1>

#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x1024_, encoding = #inverse_canonical_encoding>

!pt = !lwe.new_lwe_plaintext<application_data = <message_type = i3>, plaintext_space = #plaintext_space>

#ciphertext_space_L1_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb>
#ciphertext_space_L1_D3_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb, size = 3>
#ciphertext_space_L1_D4_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb, size = 4>

!ct = !lwe.new_lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L5_C1_>
!ct_D3 = !lwe.new_lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_D3_, key = #key, modulus_chain = #modulus_chain_L5_C1_>

!ksk_ty = tensor<10x!ct>

module {
  // CHECK: @test_relin(
  // CHECK-SAME: [[X:%.+]]: [[ct_ty:.*tensor<3x.*]],
  // CHECK-SAME: [[ksk:%.+]]: [[ksk_ty:tensor<10x.*]])
  func.func @test_relin(%x : !ct_D3, %ksk: !ksk_ty) -> !ct {
    // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
    // CHECK-DAG: [[C1:%.+]] = arith.constant 1 : index
    // CHECK-DAG: [[C2:%.+]] = arith.constant 2 : index

    // CHECK-DAG: [[extracted0:%.+]] = tensor.extract [[X]]{{\[}}[[C0]]]
    // CHECK-DAG: [[extracted1:%.+]] = tensor.extract [[X]]{{\[}}[[C1]]]
    // CHECK-DAG: [[extracted2:%.+]] = tensor.extract [[X]]{{\[}}[[C2]]]

    // CHECK-NEXT: [[constOutput:%.+]], [[linearOutput:%.+]] = polynomial.key_switch_inner [[extracted2]], [[ksk]]
    // CHECK-DAG: [[tensor0:%.+]] = polynomial.add [[extracted0]], [[constOutput]]
    // CHECK-DAG: [[tensor1:%.+]] = polynomial.add [[extracted1]], [[linearOutput]]
    // CHECK-NEXT: [[result:%.+]] = tensor.from_elements [[tensor0]], [[tensor1]]
    // CHECK-NEXT: return [[result]]
    %relin = ckks.relinearize %x, %ksk {
      from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>
    }: (!ct_D3, !ksk_ty) -> !ct
    return %relin : !ct
  }

  // func.func @test_relin(%x : tensor<!poly>, %ksk: tensor<!poly>) -> !ct {
  //   %c2 = arith.constant 2 : index
  //   %x2 = tensor.extract %x[2]
  //   %y0, %k1 = polynomial.key_switch_inner %x2, %ksk
  //   %x0 = tensor.extract %x[0]
  //   %x1 = tensor.extract %x[1]
  //   %comp0 = polynomial.add %x0, %y0
  //   %comp1 = polynomial.add %x1, %y1
  //   %result = tensor.from_elements %comp0, %comp1
  //   return %result
  // }
}
