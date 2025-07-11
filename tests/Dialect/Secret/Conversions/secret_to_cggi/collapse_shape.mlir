// RUN: heir-opt --secret-distribute-generic --secret-to-cggi %s | FileCheck %s

// Ensure that we use a collapse_shape operation to reconcile memref<2x4xlwe_ct>
// with memref<8xlwe_ct>.
module {
  // CHECK: func.func @collapse_shape
  // CHECK-SAME: (%[[arg0:.*]]: memref<2x4x[[lwe_ct:.*]]>)
  func.func @collapse_shape(%arg0: !secret.secret<memref<2xi4, strided<[?], offset: ?>>>) -> !secret.secret<memref<2xi4>> {
    %c3 = arith.constant 3 : index
    %false = arith.constant false
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    // CHECK-NOT: reinterpret_cast
    // CHECK: %[[collapse_shape:.*]] = memref.collapse_shape %[[arg0]]
    // CHECK-COUNT-6: memref.load %[[collapse_shape]]
    %0 = secret.cast %arg0 : !secret.secret<memref<2xi4, strided<[?], offset: ?>>> to !secret.secret<memref<8xi1>>
    %1 = secret.generic(%0 : !secret.secret<memref<8xi1>>) {
    ^body(%input0: memref<8xi1>):
      %3 = memref.load %input0[%c0] : memref<8xi1>
      %4 = memref.load %input0[%c1] : memref<8xi1>
      %5 = memref.load %input0[%c2] : memref<8xi1>
      %6 = memref.load %input0[%c4] : memref<8xi1>
      %7 = memref.load %input0[%c5] : memref<8xi1>
      %8 = memref.load %input0[%c6] : memref<8xi1>
      %alloc = memref.alloc() : memref<2x4xi1>
      %collapse_shape = memref.collapse_shape %alloc [[0, 1]] : memref<2x4xi1> into memref<8xi1>
      memref.store %false, %alloc[%c0, %c0] : memref<2x4xi1>
      memref.store %3, %alloc[%c0, %c1] : memref<2x4xi1>
      memref.store %4, %alloc[%c0, %c2] : memref<2x4xi1>
      memref.store %5, %alloc[%c0, %c3] : memref<2x4xi1>
      memref.store %false, %alloc[%c1, %c0] : memref<2x4xi1>
      memref.store %6, %alloc[%c1, %c1] : memref<2x4xi1>
      memref.store %7, %alloc[%c1, %c2] : memref<2x4xi1>
      memref.store %8, %alloc[%c1, %c3] : memref<2x4xi1>
      secret.yield %collapse_shape : memref<8xi1>
    } -> !secret.secret<memref<8xi1>>
    %2 = secret.cast %1 : !secret.secret<memref<8xi1>> to !secret.secret<memref<2xi4>>
    return %2 : !secret.secret<memref<2xi4>>
  }
}
