// RUN: heir-opt %s --split-preprocessing='max-return-values=16' | FileCheck %s
// TODO(#2960): remove XFAIL and update test appropriately
// XFAIL: *

// CHECK: func.func @zip_matvec__preprocessing

!Z35184371138561_i64 = !mod_arith.int<35184371138561 : i64>
!Z35184372121601_i64 = !mod_arith.int<35184372121601 : i64>
!Z36028797017456641_i64 = !mod_arith.int<36028797017456641 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L2 = !rns.rns<!Z36028797017456641_i64, !Z35184371138561_i64, !Z35184372121601_i64>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>>
#ring_rns_L2_1_x1024 = #polynomial.ring<coefficientType = !rns_L2, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L2 = #lwe.ciphertext_space<ring = #ring_rns_L2_1_x1024, encryption_type = mix>
!ct_L2 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L2, key = #key, modulus_chain = #lwe.modulus_chain<elements = <36028797017456641 : i64, 35184371138561 : i64, 35184372121601 : i64>, current = 2>>

module attributes {backend.openfhe, ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797017456641, 35184371138561, 35184372121601], P = [1152921504607338497, 1152921504608747521], logDefaultScale = 45>, scheme.ckks} {
  func.func @zip_matvec(%arg0: tensor<4x!ct_L2>, %zero_ct: !ct_L2 {client.enc_zero_arg}) -> !ct_L2 {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant dense<1.0> : tensor<4x16xf32>

    %0 = scf.for %arg1 = %c0 to %c4 step %c1 iter_args(%arg2 = %zero_ct) -> (!ct_L2) {
      %slice = tensor.extract_slice %cst[%arg1, 0] [1, 16] [1, 1] : tensor<4x16xf32> to tensor<16xf32>

      %empty = tensor.empty() : tensor<1024xf32>
      %padded = tensor.insert_slice %slice into %empty[0] [16] [1] : tensor<16xf32> into tensor<1024xf32>

      %pt = lwe.rlwe_encode %padded {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt

      %extracted_ct = tensor.extract %arg0[%arg1] : tensor<4x!ct_L2>

      %1 = ckks.mul_plain %pt, %extracted_ct : (!pt, !ct_L2) -> !ct_L2
      %2 = ckks.add %arg2, %1 : (!ct_L2, !ct_L2) -> !ct_L2

      scf.yield %2 : !ct_L2
    }
    return %0 : !ct_L2
  }
}
