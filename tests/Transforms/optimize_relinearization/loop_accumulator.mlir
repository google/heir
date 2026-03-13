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

// A nested loop where both loops do ct-ct multiplications.

// CHECK-LABEL: func.func @nested_loop_both_mul
// CHECK: secret.generic
// CHECK: affine.for
// CHECK:   affine.for
// CHECK:     arith.muli
// CHECK:     mgmt.relinearize
// CHECK:     affine.yield
// CHECK:   arith.muli
// CHECK:   mgmt.relinearize
// CHECK:   affine.yield
// CHECK: secret.yield

func.func @nested_loop_both_mul(%arg0: !secret.secret<tensor<8xi16>>) -> !secret.secret<tensor<8xi16>> {
  %0 = secret.generic(%arg0: !secret.secret<tensor<8xi16>>) {
  ^body(%input0: tensor<8xi16>):
    %outer_result = affine.for %i = 0 to 8 iter_args(%outer_acc = %input0) -> (tensor<8xi16>) {
      %inner_result = affine.for %j = 0 to 4 iter_args(%inner_acc = %outer_acc) -> (tensor<8xi16>) {
        %inner_mul = arith.muli %inner_acc, %inner_acc : tensor<8xi16>
        affine.yield %inner_mul : tensor<8xi16>
      }
      %outer_mul = arith.muli %inner_result, %inner_result : tensor<8xi16>
      affine.yield %outer_mul : tensor<8xi16>
    }
    secret.yield %outer_result : tensor<8xi16>
  } -> !secret.secret<tensor<8xi16>>
  return %0 : !secret.secret<tensor<8xi16>>
}

// A nested loop where the inner loop does ct-pt multiplications.


// CHECK-LABEL: func.func @nested_loop_inner_ct_pt
// CHECK: secret.generic
// CHECK: affine.for
// CHECK:   affine.for
// CHECK:     arith.muli
// CHECK-NOT: mgmt.relinearize
// CHECK:     affine.yield
// CHECK:   arith.muli
// CHECK:   mgmt.relinearize
// CHECK:   affine.yield
// CHECK: secret.yield

func.func @nested_loop_inner_ct_pt(%arg0: !secret.secret<tensor<8xi16>>) -> !secret.secret<tensor<8xi16>> {
  %0 = secret.generic(%arg0: !secret.secret<tensor<8xi16>>) {
  ^body(%input0: tensor<8xi16>):
    %cst = arith.constant dense<2> : tensor<8xi16>
    %outer_result = affine.for %i = 0 to 8 iter_args(%outer_acc = %input0) -> (tensor<8xi16>) {
      %inner_result = affine.for %j = 0 to 4 iter_args(%inner_acc = %outer_acc) -> (tensor<8xi16>) {
        %inner_mul = arith.muli %inner_acc, %cst : tensor<8xi16>
        affine.yield %inner_mul : tensor<8xi16>
      }
      %outer_mul = arith.muli %inner_result, %inner_result : tensor<8xi16>
      affine.yield %outer_mul : tensor<8xi16>
    }
    secret.yield %outer_result : tensor<8xi16>
  } -> !secret.secret<tensor<8xi16>>
  return %0 : !secret.secret<tensor<8xi16>>
}
