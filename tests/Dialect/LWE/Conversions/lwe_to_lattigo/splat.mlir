// RUN: heir-opt --lwe-to-lattigo %s | FileCheck %s

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
#modulus_chain_L1_C0 = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 0>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L0 = !rns.rns<!mod_arith.int<36028797018652673 : i64>>
#ring_rns_L0_1_x1024 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024, encryption_type = mix>
!ct_L0 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain_L1_C0>

module attributes {backend.lattigo, ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797018652673, 35184372121601], P = [1152921504606994433], logDefaultScale = 45, encryptionTechnique = extended>, scheme.ckks} {
  // CHECK: func.func @test_splat
  // CHECK: %[[SPLAT:.*]] = tensor.splat %{{.*}} : tensor<1x!ct>
  // CHECK: return %[[SPLAT]]
  func.func @test_splat(%ct: !ct_L0) -> tensor<1x!ct_L0> {
    %splat = tensor.splat %ct : tensor<1x!ct_L0>
    return %splat : tensor<1x!ct_L0>
  }
}
