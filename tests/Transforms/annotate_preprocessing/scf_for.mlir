// RUN: heir-opt --annotate-preprocessing %s | FileCheck %s

#poly = #polynomial.int_polynomial<x>
#pspace = #lwe.plaintext_space<
  ring = #polynomial.ring<coefficientType = i3, polynomialModulus = #poly>,
  encoding = #lwe.constant_coefficient_encoding<scaling_factor = 256>>
!plaintext = !lwe.lwe_plaintext<plaintext_space = #pspace>

// CHECK: @scf_for
func.func @scf_for(%arg0: i1, %lb: index, %ub: index, %step: index) {
  // CHECK: scf.for
  %res = scf.for %iv = %lb to %ub step %step iter_args(%sum = %arg0) -> (i1) {
    // CHECK: arith.addi
    // CHECK-SAME: {downstream_encodes = [0 : i32]}
    %0 = arith.addi %sum, %arg0 : i1
    scf.yield %0 : i1
  }
  // CHECK: lwe.encode
  // CHECK-SAME: {encode_id = 0 : i32
  %1 = lwe.encode %res { plaintext_bits = 3 : index }: i1 to !plaintext
  return
}

// CHECK: @nested_for
func.func @nested_for(%arg0: i1, %lb: index, %ub: index, %step: index) {
  // CHECK: scf.for
  %res = scf.for %iv = %lb to %ub step %step iter_args(%sum = %arg0) -> (i1) {
    // CHECK: arith.addi
    // CHECK-SAME: {downstream_encodes = [1 : i32, 2 : i32]}
    %0 = arith.addi %sum, %arg0 : i1

    // CHECK: lwe.encode
    // CHECK-SAME: {encode_id = 1 : i32
    %1 = lwe.encode %0 { plaintext_bits = 3 : index }: i1 to !plaintext

    // CHECK: scf.for
    %res2 = scf.for %iv2 = %lb to %ub step %step iter_args(%sum2 = %1) -> (!plaintext) {
      // CHECK: lwe.encode
      // CHECK-SAME: {encode_id = 2 : i32
      %2 = lwe.encode %0 { plaintext_bits = 3 : index }: i1 to !plaintext
      scf.yield %2 : !plaintext
    }

    scf.yield %0 : i1
  }
  return
}
