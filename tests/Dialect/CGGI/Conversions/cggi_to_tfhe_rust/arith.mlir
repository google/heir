// RUN: heir-opt --cggi-to-tfhe-rust %s | FileCheck %s

#key = #lwe.key<slot_index = 0>
#poly = #polynomial.int_polynomial<x>
#pspace = #lwe.plaintext_space<ring = #polynomial.ring<coefficientType = i32, polynomialModulus = #poly>, encoding = #lwe.constant_coefficient_encoding<scaling_factor = 268435456>>
#cspace = #lwe.ciphertext_space<ring = #polynomial.ring<coefficientType = i32, polynomialModulus = #poly>, encryption_type = msb, size = 742>
!ct_ty = !lwe.lwe_ciphertext<plaintext_space = #pspace, ciphertext_space = #cspace, key = #key>

// CHECK: @test_affine
// CHECK-SAME: (%[[ARG0:.*]]: !tfhe_rust.server_key, %[[ARG1:.*]]: memref<1x1x!tfhe_rust.eui32>) -> [[T:.*]] {
func.func @test_affine(%arg0: memref<1x1x!ct_ty>) -> memref<1x1x!ct_ty> {
  // CHECK: return %[[OUT:.*]] : [[T]]
  %0 = cggi.create_trivial  {value = 429 : i32} : () -> !ct_ty
  %1 = cggi.create_trivial  {value = 33 : i32} : () -> !ct_ty
  %2 = affine.load %arg0[0, 0] : memref<1x1x!ct_ty>
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x!ct_ty>
  %3 = cggi.mul %2, %1 : (!ct_ty, !ct_ty) -> !ct_ty
  %4 = cggi.add %3, %0 : (!ct_ty, !ct_ty) -> !ct_ty
  %5 = cggi.sshr %4 {shiftAmount = 2 : index} : (!ct_ty) -> !ct_ty
  affine.store %5, %alloc[0, 0] : memref<1x1x!ct_ty>
  return %alloc : memref<1x1x!ct_ty>
}
