// RUN: heir-opt --expand-copy %s | FileCheck %s

// CHECK: func.func @memref_copy()
// CHECK:      %[[ALLOC0:.*]] = memref.alloc() : memref<2x3xi32>
// CHECK-NEXT: %[[ALLOC1:.*]] = memref.alloc() : memref<2x3xi32>
// CHECK-NEXT: affine.for %{{.*}} = 0 to 2 {
// CHECK-NEXT:   affine.for %{{.*}} = 0 to 3 {
// CHECK-NEXT:     %[[LOAD:.*]] = affine.load %[[ALLOC0]]
// CHECK-NEXT:     affine.store %[[LOAD]], %[[ALLOC1]]
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: return
func.func @memref_copy() {
  %alloc = memref.alloc() : memref<2x3xi32>
  %alloc_0 = memref.alloc() : memref<2x3xi32>
  memref.copy %alloc, %alloc_0 : memref<2x3xi32> to memref<2x3xi32>
  return
}
