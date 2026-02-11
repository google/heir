// RUN: heir-opt --ckks-decompose-relinearize --ckks-decompose-keyswitch --ckks-to-lwe --lwe-to-polynomial %s

!Z1032955396097_i64 = !mod_arith.int<1032955396097 : i64>
!Z1095233372161_i64 = !mod_arith.int<1095233372161 : i64>
!Z261405424692085787_i64 = !mod_arith.int<261405424692085787 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain_L1_C1 = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64>, current = 1>
!rns_L0 = !rns.rns<!Z1095233372161_i64>
!rns_L0_1 = !rns.rns<!Z1032955396097_i64>
!rns_L1 = !rns.rns<!Z1095233372161_i64, !Z1032955396097_i64>
!rns_L2 = !rns.rns<!Z1095233372161_i64, !Z1032955396097_i64, !Z261405424692085787_i64>
#ring_rns_L1_1_x1024 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**1024>>
#ring_rns_L2_1_x1024 = #polynomial.ring<coefficientType = !rns_L2, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024, encryption_type = lsb>
#ciphertext_space_L1_D3 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024, encryption_type = lsb, size = 3>
#ciphertext_space_L2 = #lwe.ciphertext_space<ring = #ring_rns_L2_1_x1024, encryption_type = lsb>
!ct_L1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_rns_L1_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L1_C1>
!ct_L1_D3 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_rns_L1_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L1_D3, key = #key, modulus_chain = #modulus_chain_L1_C1>
!ct_L2 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_rns_L2_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L2, key = #key>
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 10, Q = [1095233372161, 1032955396097], P = [261405424692085787], logDefaultScale = 45>} {
  func.func @test_relin(%ct: !ct_L1_D3, %arg0: tensor<2x!ct_L2>) -> !ct_L1 {
    %ct_0 = ckks.relinearize %ct, %arg0 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L1_D3, tensor<2x!ct_L2>) -> !ct_L1
    return %ct_0 : !ct_L1
  }
}
