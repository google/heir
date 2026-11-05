!Zq0 = !mod_arith.int<1095233372161 : i64>
!Zq1 = !mod_arith.int<1032955396097 : i64>
!Zp0 = !mod_arith.int<998595133441 : i64>

!rns_Q = !rns.rns<!Zq0, !Zq1>
!rns_QP = !rns.rns<!Zq0, !Zq1, !Zp0>

#ring_Q = #polynomial.ring<coefficientType = !rns_Q, polynomialModulus = <1 + x**1024>>
#ring_QP = #polynomial.ring<coefficientType = !rns_QP, polynomialModulus = <1 + x**1024>>

#encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
#chain = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64>, current = 1>

#space_Q = #lwe.ciphertext_space<ring = #ring_Q, encryption_type = lsb>
#space_Q_D3 = #lwe.ciphertext_space<ring = #ring_Q, encryption_type = lsb, size = 3>
#space_QP = #lwe.ciphertext_space<ring = #ring_QP, encryption_type = lsb>

!ct = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_Q, encoding = #encoding>, ciphertext_space = #space_Q, key = #key, modulus_chain = #chain>
!ct_D3 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_Q, encoding = #encoding>, ciphertext_space = #space_Q_D3, key = #key, modulus_chain = #chain>
!ksk_ct = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_QP, encoding = #encoding>, ciphertext_space = #space_QP, key = #key>

// After --convert-polynomial-mul-to-ntt, %ksk is only used in multiplications,
// so the lowered function takes the key in NTT (eval) form.
func.func public @test_relin(%ct: !ct_D3, %ksk: tensor<2x!ksk_ct>) -> !ct {
  %0 = ckks.relinearize %ct, %ksk <from_basis = [0, 1, 2], to_basis = [0, 1]> : (!ct_D3, tensor<2x!ksk_ct>) -> !ct
  return %0 : !ct
}
