// Simple CKKS example for SimFHE demonstration
// Computes: result = (x * x) + (2 * x * y) - y

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
!pt = !lwe.new_lwe_plaintext<application_data = <message_type = f32>, plaintext_space = #plaintext_space>
#ciphertext_space_L1_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb>
#ciphertext_space_L1_D3_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb, size = 3>
!ct = !lwe.new_lwe_ciphertext<application_data = <message_type = f32>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L5_C1_>
!ct_D3 = !lwe.new_lwe_ciphertext<application_data = <message_type = f32>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_D3_, key = #key, modulus_chain = #modulus_chain_L5_C1_>

module {
  func.func @polynomial_eval(%x : !ct, %y : !ct) -> !ct {
    // Compute x^2
    %x_squared = ckks.mul %x, %x : (!ct, !ct) -> !ct_D3
    %x_squared_relin = ckks.relinearize %x_squared {
      from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>
    }: !ct_D3 -> !ct

    // Compute x * y
    %xy = ckks.mul %x, %y : (!ct, !ct) -> !ct_D3
    %xy_relin = ckks.relinearize %xy {
      from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>
    }: !ct_D3 -> !ct

    // Compute 2 * (x * y) using add
    %two_xy = ckks.add %xy_relin, %xy_relin : (!ct, !ct) -> !ct

    // Add x^2 + 2xy
    %sum1 = ckks.add %x_squared_relin, %two_xy : (!ct, !ct) -> !ct

    // Subtract y: sum1 - y
    %result = ckks.sub %sum1, %y : (!ct, !ct) -> !ct

    // Add a rotation example
    %rotated = ckks.rotate %result { offset = 1 } : !ct
    %final = ckks.add %result, %rotated : (!ct, !ct) -> !ct

    return %final : !ct
  }
}
