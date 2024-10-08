// Tests whether distribute generic operates when splitting around operations
// with regions.

// RUN: heir-opt --secret-distribute-generic="distribute-through=secret.separator" %s | FileCheck %s

// CHECK-LABEL: test_separator
// CHECK-SAME: %[[value:.*]]: !secret.secret<i32>
func.func @test_separator(
    %value: !secret.secret<i32>) -> !secret.secret<memref<10xi32>> {
  // CHECK: %[[data:.*]] = secret.generic
  // CHECK-NEXT: memref.alloc
  // CHECK:       affine.for
  // CHECK-NEXT:    affine.store
  // CHECK-NEXT:  }
  // CHECK-NEXT:    secret.yield
  // CHECK: return %[[data]]
  %c0_i32 = arith.constant 0 : i32
  %0 = secret.generic
    ins(%value : !secret.secret<i32>) {
    ^bb0(%clear_value: i32):
      %alloc_0 = memref.alloc() : memref<10xi32>
      affine.for %i = 0 to 10 {
        affine.store %c0_i32, %alloc_0[%i] : memref<10xi32>
      }
      secret.separator
      secret.yield %alloc_0 : memref<10xi32>
    } -> (!secret.secret<memref<10xi32>>)
  func.return %0 : !secret.secret<memref<10xi32>>
}

// CHECK-LABEL: test_operand_defined_in_region
// CHECK-SAME: %[[value:.*]]: !secret.secret<memref<10xi32>>
func.func @test_operand_defined_in_region(
    %value: !secret.secret<memref<10xi32>>) -> !secret.secret<memref<10xi32>> {
  // CHECK: %[[data:.*]] = secret.generic
  // CHECK-NEXT: ^bb0
  // CHECK-NEXT: memref.alloc
  // CHECK:       affine.for
  // CHECK-NEXT:    affine.load
  // CHECK-NEXT:    affine.store
  // CHECK-NEXT:  }
  // CHECK-NEXT:    secret.yield
  // CHECK: return %[[data]]
  %0 = secret.generic
    ins(%value : !secret.secret<memref<10xi32>>) {
    ^bb0(%clear_value: memref<10xi32>):
      %alloc_0 = memref.alloc() : memref<10xi32>
      affine.for %i = 0 to 10 {
        %1 = affine.load %clear_value[%i] : memref<10xi32>
        // The op operand %1 was defined within this affine.for's region and
        // does not need to be mapped to another value when moving the
        // affine.for into a new generic.
        affine.store %1, %alloc_0[%i] : memref<10xi32>
      }
      secret.separator
      secret.yield %alloc_0 : memref<10xi32>
    } -> (!secret.secret<memref<10xi32>>)
  func.return %0 : !secret.secret<memref<10xi32>>
}
