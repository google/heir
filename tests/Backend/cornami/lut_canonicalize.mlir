// RUN: heir-opt --canonicalize %s | FileCheck %s
!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>

!rns_L0_ = !rns.rns<!Z1095233372161_i64_>

#ring_Z65537_i64_1_x1024_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**1024>>
#ring_rns_L0_1_x1024_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**1024>>

#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>

#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 :
i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>

#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x1024_, encoding = #full_crt_packing_encoding>

#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024_, encryption_type = lsb>
#ciphertext_space_L0_D10_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024_, encryption_type = lsb, size = 10>

!pt_ty = !lwe.lwe_plaintext< plaintext_space = #plaintext_space>
!ct_ty = !lwe.lwe_ciphertext< plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>
!sk = !lwe.lwe_secret_key<key = #key, ring = #ring_rns_L0_1_x1024_>

func.func @require_post_pass_toposort_lut3(%arg0: tensor<8x!ct_ty>) -> !ct_ty {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %0 = tensor.extract %arg0[%c0] : tensor<8x!ct_ty>
  %1 = tensor.extract %arg0[%c1] : tensor<8x!ct_ty>
  %2 = tensor.extract %arg0[%c2] : tensor<8x!ct_ty>

  // CHECK: cggi.lut_lincomb %extracted, %extracted_0, %extracted_1 {coefficients = array<i32: 1, 2, 4>, lookup_table = 8 : ui8}
  %r1 = cggi.lut3 %0, %1, %2 {lookup_table = 8 : ui8} : !ct_ty

  return %r1 : !ct_ty
}

func.func @require_post_pass_toposort_lut2(%arg0: tensor<8x!ct_ty>) -> !ct_ty {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %0 = tensor.extract %arg0[%c0] : tensor<8x!ct_ty>
  %1 = tensor.extract %arg0[%c1] : tensor<8x!ct_ty>
  %2 = tensor.extract %arg0[%c2] : tensor<8x!ct_ty>

  // CHECK: cggi.lut_lincomb %extracted, %extracted_0 {coefficients = array<i32: 1, 2>, lookup_table = 8 : ui8}
  %r1 = cggi.lut2 %0, %1 {lookup_table = 8 : ui8} : !ct_ty

  return %r1 : !ct_ty
}
