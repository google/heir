// RUN: heir-opt %s | FileCheck %s

// CHECK: test_add_secret_and_plaintext_passed_through_block
func.func @test_add_secret_and_plaintext_passed_through_block(%value : i32) {
  %X = arith.constant 7 : i32
  %Y = secret.conceal %value : i32 -> !secret.secret<i32>
  // CHECK: secret.generic
  %Z = secret.generic
    ins(%X, %Y : i32, !secret.secret<i32>) {
    ^bb0(%x: i32, %y: i32) :
      %d = arith.addi %x, %y: i32
      secret.yield %d : i32
    } -> (!secret.secret<i32>)
  func.return
}

// CHECK: test_add_secret_and_plaintext_in_enclosing_scope
func.func @test_add_secret_and_plaintext_in_enclosing_scope(%value : i32) {
  %X = arith.constant 7 : i32
  %Y = secret.conceal %value : i32 -> !secret.secret<i32>
  // CHECK: secret.generic
  %Z = secret.generic
    ins(%Y : !secret.secret<i32>) {
    ^bb0(%y: i32) :
      %d = arith.addi %X, %y: i32
      secret.yield %d : i32
    } -> (!secret.secret<i32>)
  func.return
}

// CHECK: test_memref_store
func.func @test_memref_store(%value : i32) -> !secret.secret<memref<1xi32>> {
  %0 = secret.generic {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1xi32>
    secret.yield %alloc : memref<1xi32>
  } -> !secret.secret<memref<1xi32>>
  // CHECK: secret.generic
  affine.for %arg1 = 0 to 1 {
    secret.generic ins(%0, %arg1, %value : !secret.secret<memref<1xi32>>, index, i32) {
    ^bb0(%arg2: memref<1xi32>, %arg3: index, %arg4: i32):
      memref.store %arg4, %arg2[%arg3] : memref<1xi32>
      secret.yield
    }
  }

  func.return %0 : !secret.secret<memref<1xi32>>
}
