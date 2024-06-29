// RUN: heir-opt --secret-generic-absorb-dealloc %s | FileCheck %s

// CHECK-LABEL: test_absorb_dealloc
// CHECK-SAME: %[[Y:.*]]: !secret.secret<memref<1xi32>>) {
func.func @test_absorb_dealloc(%memref : !secret.secret<memref<1xi32>>) {
  // CHECK: %[[C0:.*]] = arith.constant 7
  %C7 = arith.constant 7 : i32
  // CHECK: %[[Z:.*]] = secret.generic ins(%[[Y]], %[[C0]] : !secret.secret<memref<1xi32>>, i32)
  %Z:2 = secret.generic ins(%memref, %C7 : !secret.secret<memref<1xi32>>, i32) {
  // CHECK-NEXT: ^[[bb0:.*]](%[[y:.*]]: memref<1xi32>, %[[c0:.*]]: i32):
  ^bb0(%y: memref<1xi32>, %c0 : i32):
    // CHECK: %[[d:.*]] = memref.alloc()
    // CHECK: memref.dealloc %[[d]]
    // CHECK: secret.yield
    affine.store %c0, %y[0] : memref<1xi32>
    %internal = memref.alloc() : memref<1xi32>
    affine.store %c0, %internal[0] : memref<1xi32>
    secret.yield %y, %internal : memref<1xi32>, memref<1xi32>
  } -> (!secret.secret<memref<1xi32>>, !secret.secret<memref<1xi32>>)
  // CHECK: secret.generic ins(%[[Z1:.*]] : !secret.secret<memref<1xi32>>)
  secret.generic ins(%Z#1 : !secret.secret<memref<1xi32>>) {
  // CHECK-NEXT: ^[[bb0:.*]](%[[y:.*]]: memref<1xi32>):
  ^bb0(%z1: memref<1xi32>):
    // CHECK-NEXT: secret.yield
    memref.dealloc %z1 : memref<1xi32>
    secret.yield
  }
  func.return
}
