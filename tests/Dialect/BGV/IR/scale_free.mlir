// RUN: heir-opt --mlir-print-local-scope %s | FileCheck %s

!Z17592186175489_i64_ = !mod_arith.int<17592186175489 : i64>
!Z33832961_i64_ = !mod_arith.int<33832961 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>

!rns_L0_ = !rns.rns<!Z33832961_i64_>
!rns_L1_ = !rns.rns<!Z33832961_i64_, !Z17592186175489_i64_>

#ring_Z65537_i64_1_x1024_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**1024>>
#ring_rns_L0_1_x1024_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**1024>>
#ring_rns_L1_1_x1024_ = #polynomial.ring<coefficientType = !rns_L1_, polynomialModulus = <1 + x**1024>>

#enc_s0 = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#enc_s2 = #lwe.full_crt_packing_encoding<scaling_factor = 2>
#enc_s5 = #lwe.full_crt_packing_encoding<scaling_factor = 5>
#enc_s7 = #lwe.full_crt_packing_encoding<scaling_factor = 7>
#key = #lwe.key<>

#modulus_chain_L1_C0_ = #lwe.modulus_chain<elements = <33832961 : i64, 17592186175489 : i64>, current = 0>
#modulus_chain_L1_C1_ = #lwe.modulus_chain<elements = <33832961 : i64, 17592186175489 : i64>, current = 1>

#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024_, encryption_type = lsb>
#ciphertext_space_L1_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb>
#ciphertext_space_L1_D3_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb, size = 3>

!ct_L0_s0 = !lwe.lwe_ciphertext<encoding = #enc_s0, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L1_C0_>
!ct_L0_s2 = !lwe.lwe_ciphertext<encoding = #enc_s2, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L1_C0_>
!ct_L1_s2 = !lwe.lwe_ciphertext<encoding = #enc_s2, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L1_C1_>
!ct_L1_s5 = !lwe.lwe_ciphertext<encoding = #enc_s5, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L1_C1_>
!ct_L1_D3_s2 = !lwe.lwe_ciphertext<encoding = #enc_s2, ciphertext_space = #ciphertext_space_L1_D3_, key = #key, modulus_chain = #modulus_chain_L1_C1_>

!pt_s7 = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_Z65537_i64_1_x1024_, encoding = #enc_s7>>

module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 13, Q = [33832961, 17592186175489], P = [17592186273793], plaintextModulus = 65537>, scheme.bgv} {
  // CHECK: @mul_and_modswitch_keep_lhs_encoding
  func.func @mul_and_modswitch_keep_lhs_encoding(%lhs: !ct_L1_s2, %rhs: !ct_L1_s5) -> !ct_L0_s2 {
    // CHECK: bgv.mul %{{.*}}, %{{.*}} : ({{.*}}) -> !lwe.lwe_ciphertext<encoding = #lwe.full_crt_packing_encoding<scaling_factor = 2>,
    %ct_0 = bgv.mul %lhs, %rhs : (!ct_L1_s2, !ct_L1_s5) -> !ct_L1_D3_s2
    %ct_1 = bgv.relinearize %ct_0 <from_basis = [0, 1, 2], to_basis = [0, 1]> : !ct_L1_D3_s2 -> !ct_L1_s2
    // CHECK: bgv.modulus_switch %{{.*}} : !lwe.lwe_ciphertext<encoding = #lwe.full_crt_packing_encoding<scaling_factor = 2>,{{.*}} -> !lwe.lwe_ciphertext<encoding = #lwe.full_crt_packing_encoding<scaling_factor = 2>,
    %ct_2 = bgv.modulus_switch %ct_1 <to_ring = #ring_rns_L0_1_x1024_> : !ct_L1_s2 -> !ct_L0_s2
    return %ct_2 : !ct_L0_s2
  }

  // CHECK: @add_plain_ignores_scale
  func.func @add_plain_ignores_scale(%ct: !ct_L0_s0, %pt: !pt_s7) -> !ct_L0_s0 {
    // CHECK: bgv.add_plain %{{.*}}, %{{.*}} : (!lwe.lwe_ciphertext<encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>,{{.*}}, !lwe.lwe_plaintext<{{.*}}scaling_factor = 7{{.*}}) -> !lwe.lwe_ciphertext<encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>,
    %0 = bgv.add_plain %ct, %pt : (!ct_L0_s0, !pt_s7) -> !ct_L0_s0
    return %0 : !ct_L0_s0
  }
}
