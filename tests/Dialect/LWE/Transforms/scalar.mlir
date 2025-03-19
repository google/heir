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

// These two types differ only on their underlying_type. The IR stays as the !ty
// for the entire computation until the final extract op.
!ty = !lwe.new_lwe_ciphertext<application_data = <message_type = i16>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

#alignment = #tensor_ext.alignment<in = [], out = [1], insertedDims = [0]>
#layout = #tensor_ext.layout<map = (d0) -> (d0), alignment = #alignment>
#original_type = #tensor_ext.original_type<originalType = i16, layout = #layout>

// encryption type is sk
module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 13, Q = [], P = [], plaintextModulus = 65537, encryptionType = sk>, scheme.bgv} {
  func.func @scalar(%arg0: !ty {tensor_ext.original_type = #original_type}) -> (!ty {tensor_ext.original_type = #original_type}) {
    %1 = bgv.add %arg0, %arg0 : (!ty, !ty) -> !ty
    return %1 : !ty
  }
}

// CHECK: @scalar
// CHECK-SAME: (%[[original_input:[^:]*]]: [[ty:[^)]*]])
// CHECK-SAME: -> [[out_ty:[^{]*]] {

// CHECK: @scalar__encrypt__arg0
// CHECK-SAME: [[arg0:%[^:]*]]: i16
                                      // 4096 = 2**13 / 2
// CHECK-NEXT: tensor.splat [[arg0]] : tensor<4096xi16>
// CHECK-NEXT: lwe.rlwe_encode

// CHECK: @scalar__decrypt__result0
// CHECK-SAME: -> i16
// CHECK:        %[[extracted:.*]] = tensor.extract
// CHECK:        return %[[extracted]]
