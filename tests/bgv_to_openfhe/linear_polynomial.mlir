// RUN: heir-opt --bgv-to-openfhe %s | FileCheck %s

!ct_ty = !lwe.rlwe_ciphertext<encoding = #lwe.polynomial_evaluation_encoding<cleartext_start = 16, cleartext_bitwidth = 16>, rlwe_params = <ring = <cmod=463187969, ideal=#polynomial.polynomial<1 + x**64>>>>
!ct_sq_ty = !lwe.rlwe_ciphertext<encoding = #lwe.polynomial_evaluation_encoding<cleartext_start = 16, cleartext_bitwidth = 16>, rlwe_params = <dimension = 3, ring = <cmod=463187969, ideal=#polynomial.polynomial<1 + x**64>>>>

// CHECK-LABEL: @linear_polynomial
// CHECK-SAME: (%[[cc:.*]]: [[cc_ty:.*crypto_context]], %[[arg0:.*]]: [[T:.*rlwe_ciphertext.*]], %[[arg1:.*]]: [[T]], %[[arg2:.*]]: [[T]], %[[arg3:.*]]: [[T]]) -> [[T]] {
func.func @linear_polynomial(%arg0: !ct_ty, %arg1: !ct_ty, %arg2: !ct_ty, %arg3: !ct_ty) -> !ct_ty {
  // CHECK: %[[v0:.*]] = openfhe.mul_no_relin %[[cc]], %[[arg0]], %[[arg2]]
  %0 = bgv.mul %arg0, %arg2  : (!ct_ty, !ct_ty) -> !ct_sq_ty
  // CHECK: %[[v1:.*]] = openfhe.relin %[[cc]], %[[v0]]
  %1 = bgv.relinearize %0  {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : !ct_sq_ty -> !ct_ty
  // CHECK: %[[v2:.*]] = openfhe.sub %[[cc]], %[[arg3]], %[[v1]]
  %2 = bgv.sub %arg3, %1  : !ct_ty
  // CHECK: %[[v3:.*]] = openfhe.sub %[[cc]], %[[v2]], %[[arg1]]
  %3 = bgv.sub %2, %arg1  : !ct_ty
  // CHECK: return %[[v3]]
  return %3 : !ct_ty
}
