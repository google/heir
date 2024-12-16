// RUN: heir-opt --openfhe-configure-crypto-context=entry-function=complex_func %s | FileCheck %s

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

!plain_ty = !lwe.new_lwe_plaintext<application_data = <message_type = tensor<32xi16>>, plaintext_space = #plaintext_space>
!ctxt_ty = !openfhe.crypto_context
!in_ty = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<32xi16>>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>
!out_ty = !lwe.new_lwe_ciphertext<application_data = <message_type = i16>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

func.func @complex_func(%arg0: !ctxt_ty, %arg1: !in_ty, %arg2: !in_ty, %cond: i1) -> !out_ty {
  %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]> : tensor<32xi16>
  %plain = lwe.rlwe_encode %cst {encoding = #full_crt_packing_encoding, ring = #ring_Z65537_i64_1_x32_} : tensor<32xi16> -> !plain_ty

  %ret = scf.if %cond -> !out_ty {
    %0 = openfhe.mul %arg0, %arg1, %arg2 : (!ctxt_ty, !in_ty, !in_ty) -> !in_ty
    %1 = openfhe.add %arg0, %0, %arg1 : (!ctxt_ty, !in_ty, !in_ty) -> !in_ty
    %2 = openfhe.mul_plain %arg0, %0, %plain : (!ctxt_ty, !in_ty, !plain_ty) -> !in_ty
    %3 = lwe.reinterpret_underlying_type %1 : !in_ty to !out_ty
    scf.yield %3 : !out_ty
  } else {
    %4 = openfhe.mul_plain %arg0, %arg1, %plain : (!ctxt_ty, !in_ty, !plain_ty) -> !in_ty
    %5 = openfhe.mul %arg0, %4, %arg2 : (!ctxt_ty, !in_ty, !in_ty) -> !in_ty
    %6 = openfhe.sub %arg0, %5, %4 : (!ctxt_ty, !in_ty, !in_ty) -> !in_ty
    %7 = openfhe.mul_plain %arg0, %5, %plain : (!ctxt_ty, !in_ty, !plain_ty) -> !in_ty
    %8 = lwe.reinterpret_underlying_type %7 : !in_ty to !out_ty
    scf.yield %8 : !out_ty
    }
  return %ret : !out_ty
}

// CHECK: @complex_func
// CHECK: @complex_func__generate_crypto_context
// CHECK: mulDepth = 3

// CHECK: @complex_func__configure_crypto_context
// CHECK: openfhe.gen_mulkey
