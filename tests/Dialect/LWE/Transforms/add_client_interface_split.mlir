// RUN: heir-opt --lwe-add-client-interface %s | FileCheck %s

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>

!rns_L0_ = !rns.rns<!Z1095233372161_i64_>

#ring_Z65537_i64_1_x32_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**32>>
#ring_rns_L0_1_x32_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**32>>

#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>

#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>

#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x32_, encoding = #full_crt_packing_encoding>

#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x32_, encryption_type = lsb>
#ciphertext_space_L0_D3_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x32_, encryption_type = lsb, size = 3>

!in_ty = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<32xi16>>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>
!mul_ty = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<32xi16>>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_D3_, key = #key, modulus_chain = #modulus_chain_L5_C0_>
!out_ty = !lwe.new_lwe_ciphertext<application_data = <message_type = i16>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 13, Q = [], P = [], plaintextModulus = 65537, encryptionType = pk>, scheme.bgv} {
  func.func @dot_product(%arg0: !in_ty, %arg1: !in_ty) -> (!out_ty, !out_ty) {
    %c7 = arith.constant 7 : index
    %0 = bgv.mul %arg0, %arg1 : (!in_ty, !in_ty) -> !mul_ty
    %1 = bgv.relinearize %0 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : !mul_ty -> !in_ty
    %2 = bgv.rotate_cols %1 { offset = 4 } : !in_ty
    %3 = bgv.add %1, %2 : (!in_ty, !in_ty) -> !in_ty
    %4 = bgv.rotate_cols %3 { offset = 2 } : !in_ty
    %5 = bgv.add %3, %4 : (!in_ty, !in_ty) -> !in_ty
    %6 = bgv.rotate_cols %5 { offset = 1 } : !in_ty
    %7 = bgv.add %5, %6 : (!in_ty, !in_ty) -> !in_ty
    %8 = bgv.extract %7, %c7 : (!in_ty, index) -> !out_ty
    return %8, %8 : !out_ty, !out_ty
  }
}

// CHECK: func.func @dot_product__encrypt__arg0
// CHECK: func.func @dot_product__encrypt__arg1
// CHECK: func.func @dot_product__decrypt__result0
// CHECK: func.func @dot_product__decrypt__result1
