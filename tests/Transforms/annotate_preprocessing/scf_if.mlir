// RUN: heir-opt --annotate-preprocessing %s | FileCheck %s

#poly = #polynomial.int_polynomial<x>
#pspace = #lwe.plaintext_space<
  ring = #polynomial.ring<coefficientType = i3, polynomialModulus = #poly>,
  encoding = #lwe.constant_coefficient_encoding<scaling_factor = 256>>
!plaintext = !lwe.lwe_plaintext<plaintext_space = #pspace>

// CHECK: @scf_if
func.func @scf_if(%cond: i1, %arg0: i1, %arg1: i1) {
  // CHECK: scf.if
  %res = scf.if %cond -> (i1) {
    // CHECK: arith.addi
    // CHECK-SAME: {downstream_encodes = [0 : i32]}
    %0 = arith.addi %arg0, %arg1 : i1
    scf.yield %0 : i1
  } else {
    // CHECK: arith.muli
    // CHECK-SAME: {downstream_encodes = [0 : i32]}
    %1 = arith.muli %arg0, %arg1 : i1
    scf.yield %1 : i1
  }
  // CHECK: lwe.encode
  // CHECK-SAME: {encode_id = 0 : i32
  %2 = lwe.encode %res { plaintext_bits = 3 : index }: i1 to !plaintext
  return
}
