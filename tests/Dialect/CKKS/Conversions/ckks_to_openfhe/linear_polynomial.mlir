// RUN: heir-opt --mlir-print-local-scope --ckks-to-lwe --lwe-to-openfhe %s | FileCheck %s

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>

!rns_L0_ = !rns.rns<!Z1095233372161_i64_>

#ring_Z65537_i64_1_x1024_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**1024>>
#ring_rns_L0_1_x1024_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**1024>>

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>

#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>

#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x1024_, encoding = #inverse_canonical_encoding>

!pt = !lwe.lwe_plaintext<application_data = <message_type = i3>, plaintext_space = #plaintext_space>

#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024_, encryption_type = lsb>
#ciphertext_space_L0_D3_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024_, encryption_type = lsb, size = 3>

!ct_ty = !lwe.lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>
!ct_sq_ty = !lwe.lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_D3_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

// CHECK: @linear_polynomial
// CHECK-SAME: (%[[cc:.*]]: [[cc_ty:!openfhe.crypto_context]], %[[arg0:.*]]: [[T:!openfhe.ciphertext]], %[[arg1:.*]]: [[T]], %[[arg2:.*]]: [[T]], %[[arg3:.*]]: [[T]]) -> [[T]]
func.func @linear_polynomial(%arg0: !ct_ty, %arg1: !ct_ty, %arg2: !ct_ty, %arg3: !ct_ty) -> !ct_ty {
  // CHECK: %[[v0:.*]] = openfhe.mul_no_relin %[[cc]], %[[arg0]], %[[arg2]]
  %0 = ckks.mul %arg0, %arg2 : (!ct_ty, !ct_ty) -> !ct_sq_ty
  // CHECK: %[[v1:.*]] = openfhe.relin %[[cc]], %[[v0]]
  %1 = ckks.relinearize %0 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_sq_ty) -> !ct_ty
  // CHECK: %[[v2:.*]] = openfhe.sub %[[cc]], %[[arg3]], %[[v1]]
  %2 = ckks.sub %arg3, %1 : (!ct_ty, !ct_ty) -> !ct_ty
  // CHECK: %[[v3:.*]] = openfhe.sub %[[cc]], %[[v2]], %[[arg1]]
  %3 = ckks.sub %2, %arg1 : (!ct_ty, !ct_ty) -> !ct_ty
  // CHECK: return %[[v3]]
  return %3 : !ct_ty
}
