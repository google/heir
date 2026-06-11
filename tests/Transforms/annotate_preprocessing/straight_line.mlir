// RUN: heir-opt --annotate-preprocessing %s | FileCheck %s

#poly = #polynomial.int_polynomial<x>
#pspace = #lwe.plaintext_space<
  ring = #polynomial.ring<coefficientType = i3, polynomialModulus = #poly>,
  encoding = #lwe.constant_coefficient_encoding<scaling_factor = 256>>
!plaintext = !lwe.lwe_plaintext<plaintext_space = #pspace>

// CHECK: @straight_line
func.func @straight_line(%arg0: i1, %arg1: i1) -> !plaintext {
  // CHECK: arith.addi %arg0, %arg1 {downstream_encodes = [0 : i32, 1 : i32]}
  %0 = arith.addi %arg0, %arg1 : i1
  // CHECK: lwe.encode %0 {encode_id = 0 : i32
  %1 = lwe.encode %0 { plaintext_bits = 3 : index }: i1 to !plaintext
  // CHECK: lwe.encode %0 {encode_id = 1 : i32
  %2 = lwe.encode %0 { plaintext_bits = 3 : index }: i1 to !plaintext
  return %1 : !plaintext
}
