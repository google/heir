// RUN: heir-opt --mlir-print-local-scope --ckks-to-lwe --lwe-to-openfhe %s | FileCheck %s

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>

!rns_L0_ = !rns.rns<!Z1095233372161_i64_>

#ring_Z65537_i64_1_x32_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**32>>
#ring_rns_L0_1_x32_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**32>>

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 1024>
#key = #lwe.key<>

#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x32_, encoding = #inverse_canonical_encoding>

!pt_i16 = !lwe.new_lwe_plaintext<application_data = <message_type = tensor<32xi16>>, plaintext_space = #plaintext_space>
!pt_i32 = !lwe.new_lwe_plaintext<application_data = <message_type = tensor<32xi32>>, plaintext_space = #plaintext_space>
!pt_i64 = !lwe.new_lwe_plaintext<application_data = <message_type = tensor<32xi64>>, plaintext_space = #plaintext_space>

!pk = !lwe.new_lwe_public_key<ring = #ring_rns_L0_1_x32_, key = #key>

//The function is adapted from the BGV form of the simple_sum.mlir test
// CHECK: @encode_i16
// CHECK-SAME: %[[cc:.*]]: !openfhe.crypto_context
// CHECK-SAME: %[[arg16:.*]]: tensor<32xi16>
func.func @encode_i16(%arg0: tensor<32xi16>, %arg1: !pk) -> !pt_i16 {
  %0 = lwe.rlwe_encode %arg0 {encoding = #inverse_canonical_encoding, ring = #ring_Z65537_i64_1_x32_} : tensor<32xi16> -> !pt_i16
  // CHECK:     %[[v0:.*]] = arith.extsi %[[arg16]] : tensor<32xi16> to tensor<32xi64>
  // CHECK:     openfhe.make_ckks_packed_plaintext %[[cc]], %[[v0]] {{.*}} tensor<32xi64>) -> !lwe.new_lwe_plaintext{{.*}} tensor<32xi16>
  // CHECK-NOT: openfhe.make_ckks_packed_plaintext {{.*}} tensor<32xi16> -> !lwe.new_lwe_plaintext{{.*}} tensor<32xi16>
  return %0 : !pt_i16
}

// CHECK: @encode_i32
// CHECK-SAME: %[[cc:.*]]: !openfhe.crypto_context
// CHECK-SAME: %[[arg0:.*]]: tensor<32xi32>
func.func @encode_i32(%arg0: tensor<32xi32>, %arg1: !pk) -> !pt_i32 {
  %0 = lwe.rlwe_encode %arg0 {encoding = #inverse_canonical_encoding, ring = #ring_Z65537_i64_1_x32_} : tensor<32xi32> -> !pt_i32
  // CHECK:     %[[v0:.*]] = arith.extsi %[[arg0]] : tensor<32xi32> to tensor<32xi64>
  // CHECK:     openfhe.make_ckks_packed_plaintext %[[cc]], %[[v0]] {{.*}} tensor<32xi64>) -> !lwe.new_lwe_plaintext{{.*}} tensor<32xi32>
  // CHECK-NOT: openfhe.make_ckks_packed_plaintext {{.*}} : tensor<32xi32> -> !lwe.new_lwe_plaintext{{.*}} tensor<32xi32>
  return %0 : !pt_i32
}

// CHECK: @encode_i64
func.func @encode_i64(%arg0: tensor<32xi64>, %arg1: !pk) -> !pt_i64 {
  %0 = lwe.rlwe_encode %arg0 {encoding = #inverse_canonical_encoding, ring = #ring_Z65537_i64_1_x32_} : tensor<32xi64> -> !pt_i64
  // CHECK:     openfhe.make_ckks_packed_plaintext {{.*}} tensor<32xi64>) -> !lwe.new_lwe_plaintext{{.*}} tensor<32xi64>
  return %0 : !pt_i64
}
