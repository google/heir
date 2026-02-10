// RUN: heir-opt --cggi-to-tfhe-rust -cse %s | FileCheck %s

#key = #lwe.key<slot_index = 0>
#poly = #polynomial.int_polynomial<x>
#pspace = #lwe.plaintext_space<ring = #polynomial.ring<coefficientType = i3, polynomialModulus = #poly>, encoding = #lwe.constant_coefficient_encoding<scaling_factor = 268435456>>
#cspace = #lwe.ciphertext_space<ring = #polynomial.ring<coefficientType = i32, polynomialModulus = #poly>, encryption_type = msb, size = 742>
!pt_ty = !lwe.lwe_plaintext<plaintext_space = #pspace>
!ct_ty = !lwe.lwe_ciphertext<plaintext_space = #pspace, ciphertext_space = #cspace, key = #key>

// CHECK: main
// CHECK-NOT: cggi
// CHECK-NOT: lwe
// CHECK: [[ALLOC:%.*]] = memref.alloc
// CHECK: memref.subview
// CHECK: memref.copy
// CHECK: return [[ALLOC]] : memref<1x1x4x!tfhe_rust.eui3>

module attributes {tf_saved_model.semantics} {
  func.func @main(%arg0: memref<1x1x8x!ct_ty>) -> memref<1x1x4x!ct_ty> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x4x!ct_ty>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        %subview = memref.subview %arg0[0, 0, 0] [1, 1, 8] [1, 1, 1] : memref<1x1x8x!ct_ty> to memref<8x!ct_ty>
        %c0 = arith.constant 0 : index
        %0 = memref.load %subview[%c0] : memref<8x!ct_ty>
        %c1 = arith.constant 1 : index
        %1 = memref.load %subview[%c1] : memref<8x!ct_ty>
        %c2 = arith.constant 2 : index
        %2 = memref.load %subview[%c2] : memref<8x!ct_ty>
        %c3 = arith.constant 3 : index
        %3 = memref.load %subview[%c3] : memref<8x!ct_ty>
        %alloc_0 = memref.alloc() : memref<4x!ct_ty>
        memref.store %3, %alloc_0[%c0] : memref<4x!ct_ty>
        memref.store %2, %alloc_0[%c1] : memref<4x!ct_ty>
        memref.store %1, %alloc_0[%c2] : memref<4x!ct_ty>
        memref.store %0, %alloc_0[%c3] : memref<4x!ct_ty>
        %subview_1 = memref.subview %alloc[0, 0, 0] [1, 1, 4] [1, 1, 1] : memref<1x1x4x!ct_ty> to memref<4x!ct_ty>
        memref.copy %alloc_0, %subview_1 : memref<4x!ct_ty> to memref<4x!ct_ty>
      }
    }
    return %alloc : memref<1x1x4x!ct_ty>
  }
}
