// RUN: heir-opt --pass-pipeline="builtin.module(func.func(affine-loop-unroll{unroll-factor=1024},canonicalize))" %s | FileCheck %s

// A test to ensure that the canonicalize pass is not slow for large secret generic bodies
// Cf. https://github.com/google/heir/issues/482

// This test should only take ~0.5s to run. Before the bug above, it took ~15s.
// It will take a bit of extra work to put a strict time limit on the bazel test runner
// via lit, so I will just leave this note here and if the test starts running slow we will
// hopefully notice it.

// CHECK-LABEL: func @fast_unrolled_loop
func.func @fast_unrolled_loop(
    %arg1 : !secret.secret<memref<1024xi32>>) -> !secret.secret<memref<1024xi32>> {
  %c5 = arith.constant 5 : i32
  %out = secret.generic ins(%arg1 : !secret.secret<memref<1024xi32>>) {
    ^bb0(%pt_arg: memref<1024xi32>):
      affine.for %i = 0 to 1024 {
        %x = memref.load %pt_arg[%i] : memref<1024xi32>
        %y = arith.addi %x, %c5 : i32
        memref.store %y, %pt_arg[%i] : memref<1024xi32>
      }
      secret.yield %pt_arg : memref<1024xi32>
    } -> !secret.secret<memref<1024xi32>>
  return %out : !secret.secret<memref<1024xi32>>
}
