// RUN: heir-opt --secret-distribute-generic="distribute-through=affine.for" %s | FileCheck %s

// CHECK: test_affine_for
// CHECK-SAME: %[[value:.*]]: !secret.secret<i32>
// CHECK-SAME: %[[data:.*]]: !secret.secret<memref<10xi32>>
func.func @test_affine_for(
    %value: !secret.secret<i32>,
    %data: !secret.secret<memref<10xi32>>) -> !secret.secret<memref<10xi32>> {
  // CHECK:       affine.for
  // CHECK:       secret.generic
  // CHECK-NEXT:    ^body
  // CHECK-NEXT:    affine.load
  // CHECK-NEXT:    arith.addi
  // CHECK-NEXT:    affine.store
  // CHECK-NEXT:    secret.yield
  // CHECK-NOT:   secret.generic
  // CHECK: return %[[data]]
  secret.generic
    (%value: !secret.secret<i32>, %data: !secret.secret<memref<10xi32>>) {
    ^body(%clear_value: i32, %clear_data : memref<10xi32>):
      affine.for %i = 0 to 10 {
        %2 = affine.load %clear_data[%i] : memref<10xi32>
        %3 = arith.addi %2, %clear_value : i32
        affine.store %3, %clear_data[%i] : memref<10xi32>
      }
      secret.yield
    } -> ()
  func.return %data : !secret.secret<memref<10xi32>>
}

// CHECK: test_affine_for_split_end
// CHECK-SAME: %[[value:.*]]: !secret.secret<i32>
// CHECK-SAME: %[[data:.*]]: !secret.secret<memref<10xi32>>
func.func @test_affine_for_split_end(
    %value: !secret.secret<i32>,
    %data: !secret.secret<memref<10xi32>>) -> !secret.secret<memref<10xi32>> {
  secret.generic
    (%value: !secret.secret<i32>, %data: !secret.secret<memref<10xi32>>) {
    ^body(%clear_value: i32, %clear_data : memref<10xi32>):
      // CHECK:    arith.constant
      // CHECK:    arith.constant
      %c7 = arith.constant 7 : i32
      %c0 = arith.constant 0 : index

      // CHECK:       secret.generic
      // CHECK-NEXT:    ^body
      // CHECK-NEXT:    memref.load
      // CHECK-NEXT:    arith.addi
      // CHECK-NEXT:    memref.store
      // CHECK-NEXT:    secret.yield
      %x = memref.load %clear_data[%c0] : memref<10xi32>
      %y = arith.addi %c7, %x : i32
      memref.store %y, %clear_data[%c0] : memref<10xi32>

      // CHECK:       affine.for
      // CHECK:       secret.generic
      // CHECK-NEXT:    ^body
      // CHECK-NEXT:    affine.load
      // CHECK-NEXT:    arith.addi
      // CHECK-NEXT:    arith.addi
      // CHECK-NEXT:    affine.store
      // CHECK-NEXT:    secret.yield
      affine.for %i = 0 to 10 {
        %2 = affine.load %clear_data[%i] : memref<10xi32>
        %3 = arith.addi %2, %clear_value : i32
        %4 = arith.addi %c7, %3 : i32
        affine.store %4, %clear_data[%i] : memref<10xi32>
      }

      secret.yield
    } -> ()
  // CHECK: return %[[data]]
  func.return %data : !secret.secret<memref<10xi32>>
}

// CHECK: test_affine_for_split_middle
// CHECK-SAME: %[[value:.*]]: !secret.secret<i32>
// CHECK-SAME: %[[data:.*]]: !secret.secret<memref<10xi32>>
func.func @test_affine_for_split_middle(
    %value: !secret.secret<i32>,
    %data: !secret.secret<memref<10xi32>>) -> !secret.secret<memref<10xi32>> {
  secret.generic
    (%value: !secret.secret<i32>, %data: !secret.secret<memref<10xi32>>) {
    ^body(%clear_value: i32, %clear_data : memref<10xi32>):
      // CHECK:    arith.constant
      // CHECK:    arith.constant
      %c7 = arith.constant 7 : i32
      %c0 = arith.constant 0 : index

      // CHECK:       secret.generic
      // CHECK-NEXT:    ^body
      // CHECK-NEXT:    memref.load
      // CHECK-NEXT:    arith.addi
      // CHECK-NEXT:    memref.store
      // CHECK-NEXT:    secret.yield
      %x = memref.load %clear_data[%c0] : memref<10xi32>
      %y = arith.addi %c7, %x : i32
      memref.store %y, %clear_data[%c0] : memref<10xi32>

      // CHECK:       affine.for
      // CHECK:       secret.generic
      // CHECK-NEXT:    ^body
      // CHECK-NEXT:    affine.load
      // CHECK-NEXT:    arith.addi
      // CHECK-NEXT:    arith.addi
      // CHECK-NEXT:    affine.store
      // CHECK-NEXT:    secret.yield
      affine.for %i = 0 to 10 {
        %2 = affine.load %clear_data[%i] : memref<10xi32>
        %3 = arith.addi %2, %clear_value : i32
        %4 = arith.addi %c7, %3 : i32
        affine.store %4, %clear_data[%i] : memref<10xi32>
      }

      // CHECK:       secret.generic
      // CHECK-NEXT:    ^body
      // CHECK-NEXT:    memref.load
      // CHECK-NEXT:    arith.addi
      // CHECK-NEXT:    memref.store
      // CHECK-NEXT:    secret.yield
      %5 = memref.load %clear_data[%c0] : memref<10xi32>
      %6 = arith.addi %y, %5 : i32
      memref.store %6, %clear_data[%c0] : memref<10xi32>

      secret.yield
    } -> ()
  // CHECK: return %[[data]]
  func.return %data : !secret.secret<memref<10xi32>>
}

// CHECK: affine_for_yielding_memref
// CHECK-SAME: %[[data:.*]]: !secret.secret<memref<10xi8>>
func.func @affine_for_yielding_memref(%arg0: !secret.secret<memref<10xi8>>) -> !secret.secret<memref<10xi8>> {
  %0 = secret.generic(%arg0 : !secret.secret<memref<10xi8>>) {
  ^body(%arg1: memref<10xi8>):
    // CHECK:       secret.generic
    // CHECK-NEXT:    memref.alloc
    // CHECK-NEXT:    secret.yield
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<10xi8>

    // CHECK:       affine.for
    // CHECK:       secret.generic
    // CHECK-NEXT:    ^body
    // CHECK-NEXT:    affine.load
    // CHECK-NEXT:    affine.store
    // CHECK-NEXT:    secret.yield
    affine.for %arg2 = 0 to 10 {
      %1 = affine.load %arg1[%arg2] : memref<10xi8>
      affine.store %1, %alloc[%arg2] : memref<10xi8>
    }
    secret.yield %alloc : memref<10xi8>
  } -> !secret.secret<memref<10xi8>>
  return %0 : !secret.secret<memref<10xi8>>
}

// CHECK: affine_for_hello_world_reproducer
// CHECK-SAME: %[[data:.*]]: !secret.secret<memref<1x80xi8>>
func.func @affine_for_hello_world_reproducer(%arg0: !secret.secret<memref<1x80xi8>>) -> !secret.secret<memref<1x80xi8>> {
  %0 = secret.generic(%arg0 : !secret.secret<memref<1x80xi8>>) {
  ^body(%arg1: memref<1x80xi8>):
    // CHECK:    arith.constant
    %c-128_i8 = arith.constant -128 : i8

    // CHECK:       secret.generic
    // CHECK-NEXT:    memref.alloc
    // CHECK-NEXT:    secret.yield
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x80xi8>

    // CHECK:       affine.for
    // CHECK-NEXT:    affine.for
    // CHECK-NEXT:      secret.generic
    // CHECK-NEXT:      ^body
    // CHECK-NEXT:      affine.load
    // CHECK-NEXT:      arith.addi
    // CHECK-NEXT:      affine.store
    // CHECK-NEXT:      secret.yield
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 80 {
        %1 = affine.load %arg1[%arg2, %arg3] : memref<1x80xi8>
        %2 = arith.addi %1, %c-128_i8 : i8
        affine.store %2, %alloc[%arg2, %arg3] : memref<1x80xi8>
      }
    }
    secret.yield %alloc : memref<1x80xi8>
  } -> !secret.secret<memref<1x80xi8>>
  return %0 : !secret.secret<memref<1x80xi8>>
}
