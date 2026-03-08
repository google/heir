// RUN: heir-opt --optimize-relinearization %s | FileCheck %s

// An accumulator loop with a ct-ct multiplication.
// The relinearize op inside the loop is essential for correctness:
// without it, the degree of %acc grows without bound across iterations.

// CHECK: func.func @loop_accumulator
// CHECK: secret.generic
// CHECK: affine.for
// CHECK: arith.muli
// CHECK: mgmt.relinearize
// CHECK: affine.yield
// CHECK: secret.yield

func.func @loop_accumulator(%arg0: !secret.secret<tensor<8xi16>>) -> !secret.secret<tensor<8xi16>> {
  %0 = secret.generic(%arg0: !secret.secret<tensor<8xi16>>) {
  ^body(%input0: tensor<8xi16>):
    %result = affine.for %i = 0 to 10 iter_args(%acc = %input0) -> (tensor<8xi16>) {
      // ct-ct multiplication: degree goes from 1 to 2
      %mul = arith.muli %acc, %acc : tensor<8xi16>
      %relin = mgmt.relinearize %mul : tensor<8xi16>
      affine.yield %relin : tensor<8xi16>
    }
    secret.yield %result : tensor<8xi16>
  } -> !secret.secret<tensor<8xi16>>
  return %0 : !secret.secret<tensor<8xi16>>
}
