// RUN: heir-opt --bgv-to-openfhe %s | FileCheck %s

#encoding_i16 = #lwe.polynomial_evaluation_encoding<cleartext_start = 16, cleartext_bitwidth = 16>
#encoding_i32 = #lwe.polynomial_evaluation_encoding<cleartext_start = 32, cleartext_bitwidth = 32>
#encoding_i64 = #lwe.polynomial_evaluation_encoding<cleartext_start = 64, cleartext_bitwidth = 64>

#my_poly = #polynomial.int_polynomial<1 + x**32>

#ring = #polynomial.ring<coefficientType = i32, coefficientModulus = 463187969 : i32, polynomialModulus = #my_poly>
#params = #lwe.rlwe_params<ring = #ring>

!pt_i16 = !lwe.rlwe_plaintext<encoding = #encoding_i16, ring = #ring, underlying_type = tensor<32xi16>>
!pt_i32 = !lwe.rlwe_plaintext<encoding = #encoding_i32, ring = #ring, underlying_type = tensor<32xi32>>
!pt_i64 = !lwe.rlwe_plaintext<encoding = #encoding_i64, ring = #ring, underlying_type = tensor<32xi64>>
!pt_scalar = !lwe.rlwe_plaintext<encoding = #encoding_i64, ring = #ring, underlying_type = i64>

!pk = !lwe.rlwe_public_key<rlwe_params = #params>

//The function is adapted from the BGV form of the simple_sum.mlir test
// CHECK-LABEL: @encode_i16
// CHECK-SAME: %[[cc:.*]]: !openfhe.crypto_context
// CHECK-SAME: %[[arg16:.*]]: tensor<32xi16>
func.func @encode_i16(%arg0: tensor<32xi16>, %arg1: !pk) -> !pt_i16 {
  %0 = lwe.rlwe_encode %arg0 {encoding = #encoding_i16, ring = #ring} : tensor<32xi16> -> !pt_i16
  // CHECK:     %[[v0:.*]] = arith.extsi %[[arg16]] : tensor<32xi16> to tensor<32xi64>
  // CHECK:     openfhe.make_packed_plaintext %[[cc]], %[[v0]] {{.*}} tensor<32xi64>) -> !lwe.rlwe_plaintext{{.*}} tensor<32xi16>>
  // CHECK-NOT: openfhe.make_packed_plaintext {{.*}} tensor<32xi16> -> !lwe.rlwe_plaintext{{.*}} tensor<32xi16>>
  return %0 : !pt_i16
}

// CHECK-LABEL: @encode_i32
// CHECK-SAME: %[[cc:.*]]: !openfhe.crypto_context
// CHECK-SAME: %[[arg0:.*]]: tensor<32xi32>
func.func @encode_i32(%arg0: tensor<32xi32>, %arg1: !pk) -> !pt_i32 {
  %0 = lwe.rlwe_encode %arg0 {encoding = #encoding_i32, ring = #ring} : tensor<32xi32> -> !pt_i32
  // CHECK:     %[[v0:.*]] = arith.extsi %[[arg0]] : tensor<32xi32> to tensor<32xi64>
  // CHECK:     openfhe.make_packed_plaintext %[[cc]], %[[v0]] {{.*}} tensor<32xi64>) -> !lwe.rlwe_plaintext{{.*}} tensor<32xi32>>
  // CHECK-NOT: openfhe.make_packed_plaintext {{.*}} : tensor<32xi32> -> !lwe.rlwe_plaintext{{.*}} tensor<32xi32>>
  return %0 : !pt_i32
}

// CHECK-LABEL: @encode_i64
func.func @encode_i64(%arg0: tensor<32xi64>, %arg1: !pk) -> !pt_i64 {
  %0 = lwe.rlwe_encode %arg0 {encoding = #encoding_i64, ring = #ring} : tensor<32xi64> -> !pt_i64
  // CHECK:     openfhe.make_packed_plaintext {{.*}} tensor<32xi64>) -> !lwe.rlwe_plaintext{{.*}} tensor<32xi64>>
  return %0 : !pt_i64
}

// CHECK-LABEL: @encode_scalar
// CHECK-SAME: %[[cc:.*]]: !openfhe.crypto_context
// CHECK-SAME: %[[arg0:.*]]: i64
func.func @encode_scalar(%arg0: i64, %arg1: !pk) -> !pt_scalar {
  %0 = lwe.rlwe_encode %arg0 {encoding = #encoding_i64, ring = #ring} : i64 -> !pt_scalar
  // CHECK:     %[[v0:.*]] = tensor.splat
  // CHECK:     openfhe.make_packed_plaintext %[[cc]], %[[v0]] {{.*}} tensor<32xi64>) -> !lwe.rlwe_plaintext{{.*}} i64
  return %0 : !pt_scalar
}
