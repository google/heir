// RUN: heir-opt --secret-extract-generic-body --split-input-file %s | FileCheck %s

// CHECK: test_add
// CHECK-SAME:     %[[ARG:.*]]: i32
// CHECK-SAME:  ) {
// CHECK-NEXT:    secret.conceal %[[ARG]]
// CHECK-NEXT:    secret.generic
// CHECK-NEXT:      ^body(%[[SARG:.*]]: i32)
// CHECK-NEXT:      %[[V0:.*]] = func.call [[F0:.*]](%[[SARG]]) : (i32) -> i32
// CHECK-NEXT:      secret.yield %[[V0]]
// CHECK:         return
// CHECK:       }
module {
  func.func @test_add(%value : i32) {
    %Y = secret.conceal %value : i32 -> !secret.secret<i32>
    %Z = secret.generic(%Y : !secret.secret<i32>) {
      ^body(%y: i32) :
        %c7_i32_0 = arith.constant 7 : i32
        %d = arith.addi %c7_i32_0, %y: i32
        secret.yield %d : i32
      } -> (!secret.secret<i32>)
    func.return
  }
}

// -----

// CHECK: test_multiple
// CHECK-SAME:     %[[ARG:.*]]: i32
// CHECK-SAME:  ) {
// CHECK-NEXT:    secret.conceal %[[ARG]]
// CHECK-NEXT:    secret.generic
// CHECK-NEXT:      ^body(%[[SARG:.*]]: i32)
// CHECK-NEXT:      %[[V0:.*]] = func.call [[F1:.*]](%[[SARG]]) : (i32) -> i32
// CHECK-NEXT:      secret.yield %[[V0]]
// CHECK:    secret.generic
// CHECK-NEXT:      ^body(%[[SARG1:.*]]: i32)
// CHECK-NEXT:      %[[V1:.*]] = func.call [[F2:.*]](%[[SARG1]]) : (i32) -> i32
// CHECK-NEXT:      secret.yield %[[V1]]
// CHECK:         return
// CHECK:       }
module {
  func.func @test_multiple(%value : i32) {
    %Y = secret.conceal %value : i32 -> !secret.secret<i32>
    %Z = secret.generic(%Y : !secret.secret<i32>) {
      ^body(%y: i32) :
        %c7_i32_0 = arith.constant 7 : i32
        %d = arith.addi %c7_i32_0, %y: i32
        secret.yield %d : i32
      } -> (!secret.secret<i32>)
    %A = secret.generic(%Z : !secret.secret<i32>) {
      ^body(%y: i32) :
        %c12_i32_0 = arith.constant 12 : i32
        %d = arith.addi %c12_i32_0, %y: i32
        secret.yield %d : i32
      } -> (!secret.secret<i32>)
    func.return
  }
}

// -----

// CHECK: module

// CHECK: generic_
// CHECK-SAME:    %[[GARG:.*]]: memref<1xi32>
// CHECK:         affine.for
// CHECK-NEXT:      affine.load
// CHECK-NEXT:      arith.addi
// CHECK-NEXT:      affine.store
// CHECK:      return

// CHECK: test_region
// CHECK-SAME:     %[[ARG:.*]]: memref<1xi32>
// CHECK-SAME:  ) {
// CHECK-NEXT:    secret.conceal %[[ARG]]
// CHECK-NEXT:    secret.generic
// CHECK-NEXT:      ^body(%[[SARG:.*]]: memref<1xi32>)
// CHECK-NEXT:      %[[V0:.*]] = func.call [[F1:.*]](%[[SARG]]) : (memref<1xi32>) -> memref<1xi32>
// CHECK-NEXT:      secret.yield %[[V0]]
// CHECK:         return
// CHECK:       }
module {
  func.func @test_region(%value : memref<1xi32>) {
    %Y = secret.conceal %value : memref<1xi32> -> !secret.secret<memref<1xi32>>
    %Z = secret.generic(%Y : !secret.secret<memref<1xi32>>) {
      ^body(%y: memref<1xi32>) :
        %c7_i32_0 = arith.constant 7 : i32
        affine.for %i = 0 to 1 {
          %0 = affine.load %y[%i] : memref<1xi32>
          %1 = arith.addi %c7_i32_0, %0: i32
          affine.store %1, %y[%i] : memref<1xi32>
        }
        secret.yield %y : memref<1xi32>
      } -> (!secret.secret<memref<1xi32>>)
    func.return
  }
}
