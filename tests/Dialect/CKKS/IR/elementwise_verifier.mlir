// RUN: heir-opt --verify-diagnostics --split-input-file %s

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z1032955396097_i64_ = !mod_arith.int<1032955396097 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>
!rns_L1_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_>
#ring_Z65537_i64_1_x1024_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**1024>>
#ring_rns_L1_1_x1024_ = #polynomial.ring<coefficientType = !rns_L1_, polynomialModulus = <1 + x**1024>>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain_L5_C1_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 1>
#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x1024_, encoding = #inverse_canonical_encoding>
#ciphertext_space_L1_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb>
#ciphertext_space_L1_D3_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb, size = 3>
!ct = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L5_C1_>
!ct_D3 = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_D3_, key = #key, modulus_chain = #modulus_chain_L5_C1_>

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797019389953, 35184372121601, 35184372744193, 35184373006337, 35184373989377, 35184374874113], P = [36028797019488257, 36028797020209153], logDefaultScale = 45>} {
  func.func @test_relinearize_elementwise(%ct : tensor<10x!ct_D3>) -> tensor<10x!ct> {
    %out = ckks.relinearize %ct {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (tensor<10x!ct_D3>) -> tensor<10x!ct>
    return %out : tensor<10x!ct>
  }
}

// -----

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z1032955396097_i64_ = !mod_arith.int<1032955396097 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>
!rns_L1_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_>
#ring_Z65537_i64_1_x1024_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**1024>>
#ring_rns_L1_1_x1024_ = #polynomial.ring<coefficientType = !rns_L1_, polynomialModulus = <1 + x**1024>>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain_L5_C1_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 1>
#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x1024_, encoding = #inverse_canonical_encoding>
#ciphertext_space_L1_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb>
#ciphertext_space_L1_D3_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb, size = 3>
!ct = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L5_C1_>
!ct_D3 = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_D3_, key = #key, modulus_chain = #modulus_chain_L5_C1_>

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797019389953, 35184372121601, 35184372744193, 35184373006337, 35184373989377, 35184374874113], P = [36028797019488257, 36028797020209153], logDefaultScale = 45>} {
  func.func @test_relinearize_elementwise_error(%ct : tensor<10x!ct_D3>) -> tensor<11x!ct> {
    // expected-error@+1 {{expected all tensor results to have the same shape as mappable operands, but found shape (10) at operand 0 and shape (11) at result 0}}
    %out = ckks.relinearize %ct {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (tensor<10x!ct_D3>) -> tensor<11x!ct>
    return %out : tensor<11x!ct>
  }
}
