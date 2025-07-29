// RUN: heir-opt --openfhe-configure-crypto-context=entry-function=simple_sum %s | FileCheck %s

!Z1032955396097_i64_ = !mod_arith.int<1032955396097 : i64>
!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>
#modulus_chain_L5_C1_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 1>
!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
!rns_L1_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_>
#ring_Z65537_i64_1_x32_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**32>>
#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x32_, encoding = #full_crt_packing_encoding>
#ring_rns_L0_1_x32_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**32>>
#ring_rns_L1_1_x32_ = #polynomial.ring<coefficientType = !rns_L1_, polynomialModulus = <1 + x**32>>
!pt = !lwe.lwe_plaintext<application_data = <message_type = tensor<32xi16>>, plaintext_space = #plaintext_space>
!pt1 = !lwe.lwe_plaintext<application_data = <message_type = i16>, plaintext_space = #plaintext_space>
#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x32_, encryption_type = lsb>
#ciphertext_space_L1_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x32_, encryption_type = lsb>
!ct_L0_ = !lwe.lwe_ciphertext<application_data = <message_type = i16>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>
!ct_L1_ = !lwe.lwe_ciphertext<application_data = <message_type = tensor<32xi16>>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L5_C1_>
!ct_L1_1 = !lwe.lwe_ciphertext<application_data = <message_type = i16>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L5_C1_>

func.func @simple_sum(%arg0: !openfhe.crypto_context, %arg1: !ct_L1_) -> !ct_L0_ {
  %cst = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]> : tensor<32xi64>
  %0 = openfhe.rot %arg0, %arg1 {index = 16 : index} : (!openfhe.crypto_context, !ct_L1_) -> !ct_L1_
  %1 = openfhe.add %arg0, %arg1, %0 : (!openfhe.crypto_context, !ct_L1_, !ct_L1_) -> !ct_L1_
  %2 = openfhe.rot %arg0, %1 {index = 8 : index} : (!openfhe.crypto_context, !ct_L1_) -> !ct_L1_
  %3 = openfhe.add %arg0, %1, %2 : (!openfhe.crypto_context, !ct_L1_, !ct_L1_) -> !ct_L1_
  %4 = openfhe.rot %arg0, %3 {index = 4 : index} : (!openfhe.crypto_context, !ct_L1_) -> !ct_L1_
  %5 = openfhe.add %arg0, %3, %4 : (!openfhe.crypto_context, !ct_L1_, !ct_L1_) -> !ct_L1_
  %6 = openfhe.rot %arg0, %5 {index = 2 : index} : (!openfhe.crypto_context, !ct_L1_) -> !ct_L1_
  %7 = openfhe.add %arg0, %5, %6 : (!openfhe.crypto_context, !ct_L1_, !ct_L1_) -> !ct_L1_
  %8 = openfhe.rot %arg0, %7 {index = 1 : index} : (!openfhe.crypto_context, !ct_L1_) -> !ct_L1_
  %9 = openfhe.add %arg0, %7, %8 : (!openfhe.crypto_context, !ct_L1_, !ct_L1_) -> !ct_L1_
  %10 = openfhe.make_packed_plaintext %arg0, %cst : (!openfhe.crypto_context, tensor<32xi64>) -> !pt
  %11 = openfhe.mul_plain %arg0, %9, %10 : (!openfhe.crypto_context, !ct_L1_, !pt) -> !ct_L1_
  %12 = openfhe.rot %arg0, %11 {index = 31 : index} : (!openfhe.crypto_context, !ct_L1_) -> !ct_L1_
  %13 = lwe.reinterpret_application_data %12 : !ct_L1_ to !ct_L1_1
  %14 = openfhe.mod_reduce %arg0, %13 : (!openfhe.crypto_context, !ct_L1_1) -> !ct_L0_
  return %14 : !ct_L0_
}

// CHECK: @simple_sum
// CHECK: @simple_sum__generate_crypto_context
// CHECK: mulDepth = 1

// CHECK: @simple_sum__configure_crypto_context
// CHECK: openfhe.gen_rotkey
