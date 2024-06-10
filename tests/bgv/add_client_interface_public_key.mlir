// RUN: heir-opt --bgv-add-client-interface=use-public-key=true %s | FileCheck %s

// These two types differ only on their underlying_type. The IR stays as the !in_ty
// for the entire computation until the final extract op.
#encoding = #lwe.polynomial_evaluation_encoding<cleartext_start = 16, cleartext_bitwidth = 16>
#params = #lwe.rlwe_params<ring = <coefficientType = i32, coefficientModulus = 463187969 : i32, polynomialModulus=#polynomial.int_polynomial<1 + x**32>>>
!in_ty = !lwe.rlwe_ciphertext<encoding = #encoding, rlwe_params = #params, underlying_type = tensor<32xi16>>
!out_ty = !lwe.rlwe_ciphertext<encoding = #encoding, rlwe_params = #params, underlying_type = i16>

func.func @simple_sum(%arg0: !in_ty) -> !out_ty {
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %c31 = arith.constant 31 : index
  %0 = bgv.rotate %arg0, %c16 : !in_ty, index
  %1 = bgv.add %arg0, %0 : !in_ty
  %2 = bgv.rotate %1, %c8 : !in_ty, index
  %3 = bgv.add %1, %2 : !in_ty
  %4 = bgv.rotate %3, %c4 : !in_ty, index
  %5 = bgv.add %3, %4 : !in_ty
  %6 = bgv.rotate %5, %c2 : !in_ty, index
  %7 = bgv.add %5, %6 : !in_ty
  %8 = bgv.rotate %7, %c1 : !in_ty, index
  %9 = bgv.add %7, %8 : !in_ty
  %10 = bgv.extract %9, %c31 : (!in_ty, index) -> !out_ty
  return %10 : !out_ty
}

// CHECK-LABEL: @simple_sum
// CHECK-SAME: (%[[original_input:[^:]*]]: [[in_ty:[^)]*]])
// CHECK-SAME: -> [[out_ty:[^{]*]] {

// CHECK: @simple_sum__encrypt
// CHECK-SAME: (%[[arg0:[^:]*]]: tensor<32xi16>,
// CHECK-SAME:     %[[sk:.*]]: !lwe.rlwe_public_key
// CHECK-SAME:     -> [[in_ty]] {
// CHECK-NEXT:   %[[encoded:.*]] = lwe.rlwe_encode %[[arg0]]
// CHECK-NEXT:   %[[encrypted:.*]] = bgv.encrypt %[[encoded]], %[[sk]]
// CHECK-NEXT:   return %[[encrypted]]

// CHECK: @simple_sum__decrypt
// CHECK-SAME: (%[[arg1:[^:]*]]: [[out_ty]]
// CHECK-SAME:     %[[sk2:.*]]: !lwe.rlwe_secret_key
// CHECK-SAME:     -> i16 {
// CHECK-NEXT:   %[[decrypted:.*]] = bgv.decrypt %[[arg1]], %[[sk2]]
// CHECK-NEXT:   %[[decoded:.*]] = lwe.rlwe_decode %[[decrypted]]
// CHECK-NEXT:   return %[[decoded]]
