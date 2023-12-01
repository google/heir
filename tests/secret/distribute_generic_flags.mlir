// RUN: heir-opt --secret-distribute-generic="distribute-through=affine.for" %s | FileCheck %s

// CHECK-LABEL: test_affine_for
// CHECK-SAME: %[[value:.*]]: !secret.secret<i32>
// CHECK-SAME: %[[data:.*]]: !secret.secret<memref<10xi32>>
func.func @test_affine_for(
    %value: !secret.secret<i32>,
    %data: !secret.secret<memref<10xi32>>) -> !secret.secret<memref<10xi32>> {
  // CHECK:       affine.for
  // CHECK:       secret.generic
  // CHECK-NEXT:    bb
  // CHECK-NEXT:    affine.load
  // CHECK-NEXT:    arith.addi
  // CHECK-NEXT:    affine.store
  // CHECK-NEXT:    secret.yield
  // CHECK-NOT:   secret.generic
  // CHECK: return %[[data]]
  secret.generic
    ins(%value, %data : !secret.secret<i32>, !secret.secret<memref<10xi32>>) {
    ^bb0(%clear_value: i32, %clear_data : memref<10xi32>):
      affine.for %i = 0 to 10 {
        %2 = affine.load %clear_data[%i] : memref<10xi32>
        %3 = arith.addi %2, %clear_value : i32
        affine.store %3, %clear_data[%i] : memref<10xi32>
      }
      secret.yield
    } -> ()
  func.return %data : !secret.secret<memref<10xi32>>
}
