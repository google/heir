// RUN: heir-opt --openfhe-configure-crypto-context=entry-function=simple_sum %s | FileCheck %s

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

func.func @simple_sum(%arg0: !ctxt_ty, %arg1: !in_ty) -> !out_ty {
  %c31_i64 = arith.constant 31 : i64
  %c1_i64 = arith.constant 1 : i64
  %c2_i64 = arith.constant 2 : i64
  %c4_i64 = arith.constant 4 : i64
  %c8_i64 = arith.constant 8 : i64
  %cst = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]> : tensor<32xi16>
  %c16_i64 = arith.constant 16 : i64
  %0 = openfhe.rot %arg0, %arg1 { index = 16 } : (!ctxt_ty, !in_ty) -> !in_ty
  %1 = openfhe.add %arg0, %arg1, %0 : (!ctxt_ty, !in_ty, !in_ty) -> !in_ty
  %2 = openfhe.rot %arg0, %1 { index = 8 } : (!ctxt_ty, !in_ty) -> !in_ty
  %3 = openfhe.add %arg0, %1, %2 : (!ctxt_ty, !in_ty, !in_ty) -> !in_ty
  %4 = openfhe.rot %arg0, %3 { index = 4 } : (!ctxt_ty, !in_ty) -> !in_ty
  %5 = openfhe.add %arg0, %3, %4 : (!ctxt_ty, !in_ty, !in_ty) -> !in_ty
  %6 = openfhe.rot %arg0, %5 { index = 2 } : (!ctxt_ty, !in_ty) -> !in_ty
  %7 = openfhe.add %arg0, %5, %6 : (!ctxt_ty, !in_ty, !in_ty) -> !in_ty
  %8 = openfhe.rot %arg0, %7 { index = 1 } : (!ctxt_ty, !in_ty) -> !in_ty
  %9 = openfhe.add %arg0, %7, %8 : (!ctxt_ty, !in_ty, !in_ty) -> !in_ty
  %10 = lwe.rlwe_encode %cst {encoding = #full_crt_packing_encoding, ring = #ring_Z65537_i64_1_x32_} : tensor<32xi16> -> !plain_ty
  %11 = openfhe.mul_plain %arg0, %9, %10 : (!ctxt_ty, !in_ty, !plain_ty) -> !in_ty
  %12 = openfhe.rot %arg0, %11 { index = 31 } : (!ctxt_ty, !in_ty) -> !in_ty
  %13 = lwe.reinterpret_underlying_type %12 : !in_ty to !out_ty
  return %13 : !out_ty
}

// CHECK: @simple_sum
// CHECK: @simple_sum__generate_crypto_context
// CHECK: mulDepth = 1

// CHECK: @simple_sum__configure_crypto_context
// CHECK: openfhe.gen_mulkey
// CHECK: openfhe.gen_rotkey
