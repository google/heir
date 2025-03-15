// RUN: heir-opt --mlir-print-local-scope --lwe-add-client-interface %s | FileCheck %s

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

// These two types differ only on their underlying_type. The IR stays as the !in_ty
// for the entire computation until the final extract op.
!in_ty = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<32xi16>>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>
!out_ty = !lwe.new_lwe_ciphertext<application_data = <message_type = i16>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

// encryption type is sk
module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 13, Q = [], P = [], plaintextModulus = 65537, encryptionType = sk>, scheme.bgv} {
  func.func @simple_sum(%arg0: !in_ty) -> !out_ty {
    %c31 = arith.constant 31 : index
    %0 = bgv.rotate_cols %arg0 { offset = 16 } : !in_ty
    %1 = bgv.add %arg0, %0 : (!in_ty, !in_ty) -> !in_ty
    %2 = bgv.rotate_cols %1 { offset = 8 } : !in_ty
    %3 = bgv.add %1, %2 : (!in_ty, !in_ty) -> !in_ty
    %4 = bgv.rotate_cols %3 { offset = 4 } : !in_ty
    %5 = bgv.add %3, %4 : (!in_ty, !in_ty) -> !in_ty
    %6 = bgv.rotate_cols %5 { offset = 2 } : !in_ty
    %7 = bgv.add %5, %6 : (!in_ty, !in_ty) -> !in_ty
    %8 = bgv.rotate_cols %7 { offset = 1 } : !in_ty
    %9 = bgv.add %7, %8 : (!in_ty, !in_ty) -> !in_ty
    %10 = bgv.extract %9, %c31 : (!in_ty, index) -> !out_ty
    return %10 : !out_ty
  }
}

// CHECK-LABEL: @simple_sum
// CHECK-SAME: (%[[original_input:[^:]*]]: [[in_ty:[^)]*]])
// CHECK-SAME: -> [[out_ty:[^{]*]] {

// CHECK: @simple_sum__encrypt
// CHECK-SAME: (%[[arg0:[^:]*]]: tensor<32xi16>,
// CHECK-SAME:     %[[sk:.*]]: !lwe.new_lwe_secret_key
// CHECK-SAME:     -> [[in_ty]] {
// CHECK-NEXT:   %[[encoded:.*]] = lwe.rlwe_encode %[[arg0]]
// CHECK-NEXT:   %[[encrypted:.*]] = lwe.rlwe_encrypt %[[encoded]], %[[sk]]
// CHECK-NEXT:   return %[[encrypted]]

// CHECK: @simple_sum__decrypt
// CHECK-SAME: (%[[arg1:[^:]*]]: [[out_ty]]
// CHECK-SAME:     %[[sk2:.*]]: !lwe.new_lwe_secret_key
// CHECK-SAME:     -> i16 {
// CHECK-NEXT:   %[[decrypted:.*]] = lwe.rlwe_decrypt %[[arg1]], %[[sk2]]
// CHECK-NEXT:   %[[decoded:.*]] = lwe.rlwe_decode %[[decrypted]]
// CHECK-NEXT:   return %[[decoded]]
