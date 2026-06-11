// RUN: heir-opt --annotate-preprocessing %s | FileCheck %s

#poly = #polynomial.int_polynomial<x>
#pspace = #lwe.plaintext_space<
  ring = #polynomial.ring<coefficientType = i3, polynomialModulus = #poly>,
  encoding = #lwe.constant_coefficient_encoding<scaling_factor = 256>>
!plaintext = !lwe.lwe_plaintext<plaintext_space = #pspace>

// CHECK: @affine_loops
func.func @affine_loops(%arg0: memref<10xi1>) {
  affine.for %i = 0 to 10 {
    // CHECK: affine.load
    // CHECK-SAME: {downstream_encodes = [0 : i32]}
    %0 = affine.load %arg0[%i] : memref<10xi1>
    // CHECK: lwe.encode
    // CHECK-SAME: {encode_id = 0 : i32
    %1 = lwe.encode %0 { plaintext_bits = 3 : index }: i1 to !plaintext
  }
  return
}
