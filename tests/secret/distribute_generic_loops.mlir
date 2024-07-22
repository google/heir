// RUN: heir-opt --wrap-generic --secret-distribute-generic="distribute-through=affine.for,scf.for" --split-input-file %s | FileCheck %s

// CHECK-LABEL: simple_sum
// CHECK-SAME: %[[data:.*]]: !secret.secret<tensor<32xi16>>
// CHECK-SAME: -> !secret.secret<i16>
func.func @simple_sum(%arg0: tensor<32xi16> { secret.secret }) -> i16 {
  // CHECK: %[[c0:.*]] = arith.constant 0 : i16
  // CHECK: %[[secret_c0:.*]] = secret.conceal %[[c0]]
  %c0 = arith.constant 0 : index
  %c0_si16 = arith.constant 0 : i16
  // CHECK: %[[sum:.*]] = affine.for
  // CHECK-SAME: iter_args(%[[sum_iter:.*]] = %[[secret_c0]])
  // CHECK-SAME: -> (!secret.secret<i16>)
  %0 = affine.for %i = 0 to 32 iter_args(%sum_iter = %c0_si16) -> i16 {
    // CHECK-NEXT: %[[data_i:.*]] = secret.generic ins(%[[data]], %[[sum_iter]]
    %1 = tensor.extract %arg0[%i] : tensor<32xi16>
    %2 = arith.addi %1, %sum_iter : i16
    // CHECK: secret.yield
    // CHECK-NEXT: -> !secret.secret<i16>
    // CHECK-NEXT: affine.yield %[[data_i]] : !secret.secret<i16>
    affine.yield %2 : i16
  }
  // CHECK: return %[[sum]] : !secret.secret<i16>
  return %0 : i16
}

// -----

// CHECK-LABEL: simple_sum
// CHECK-SAME: %[[data:.*]]: !secret.secret<tensor<32xi16>>
// CHECK-SAME: -> !secret.secret<i16>
func.func @simple_sum(%arg0: tensor<32xi16> { secret.secret }) -> i16 {
  // CHECK: %[[c0:.*]] = arith.constant 0 : i16
  // CHECK: %[[secret_c0:.*]] = secret.conceal %[[c0]]
  %c0 = arith.constant 0 : index
  %c0_si16 = arith.constant 0 : i16
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[sum:.*]] = scf.for
  // CHECK-SAME: iter_args(%[[sum_iter:.*]] = %[[secret_c0]])
  // CHECK-SAME: -> (!secret.secret<i16>)
  %0 = scf.for %i = %c0 to %c32 step %c1 iter_args(%sum_iter = %c0_si16) -> i16 {
    // CHECK-NEXT: %[[data_i:.*]] = secret.generic ins(%[[data]], %[[sum_iter]]
    %1 = tensor.extract %arg0[%i] : tensor<32xi16>
    %2 = arith.addi %1, %sum_iter : i16
    // CHECK: secret.yield
    // CHECK-NEXT: -> !secret.secret<i16>
    // CHECK-NEXT: scf.yield %[[data_i]] : !secret.secret<i16>
    scf.yield %2 : i16
  }
  // CHECK: return %[[sum]] : !secret.secret<i16>
  return %0 : i16
}

// -----

// CHECK-LABEL: simple_sum_bound
// CHECK-SAME: %[[data:.*]]: !secret.secret<tensor<32xi16>>
// CHECK-SAME: %[[arg1:.*]]: index
// CHECK-SAME: -> !secret.secret<i16>
func.func @simple_sum_bound(%arg0: tensor<32xi16> { secret.secret }, %arg1: index) -> i16 {
    %c0 = arith.constant 0 : index
    %c0_i16 = arith.constant 0 : i16
    %c1 = arith.constant 1 : index
  // CHECK: %[[sum:.*]] = scf.for
  // CHECK-SAME: to %[[arg1]]
  // CHECK-SAME: -> (!secret.secret<i16>)
    %1 = scf.for %arg4 = %c0 to %arg1 step %c1 iter_args(%arg5 = %c0_i16) -> (i16) {
      // CHECK-NEXT: secret.generic
        %extracted = tensor.extract %arg0[%arg4] : tensor<32xi16>
        %2 = arith.addi %extracted, %arg5 : i16
        scf.yield %2 : i16
    }
    // CHECK: return
    return %1 : i16
}

// -----

// Tests a loop with a secret bound - in this case the distribute pattern does
// not match and the generic is not distributed through the region.

// CHECK-LABEL: simple_sum_secret_bound
// CHECK-SAME: %[[data:.*]]: !secret.secret<tensor<32xi16>>
// CHECK-SAME: %[[arg1:.*]]: !secret.secret<index>
// CHECK-SAME: -> !secret.secret<i16>
func.func @simple_sum_secret_bound(%arg0: tensor<32xi16> { secret.secret }, %arg1: index { secret.secret }) -> i16 {
    %c0 = arith.constant 0 : index
    %c0_i16 = arith.constant 0 : i16
    %c1 = arith.constant 1 : index
    // CHECK: secret.generic ins(%[[data]], %[[arg1]] : !secret.secret<tensor<32xi16>>, !secret.secret<index>)
    // CHECK-NEXT: ^bb0(%[[DATA:.*]]: tensor<32xi16>, %[[ARG1:.*]]: index):
    // CHECK-NEXT:   %[[sum:.*]] = scf.for
    // CHECK-SAME: to %[[ARG1]]
    %1 = scf.for %arg4 = %c0 to %arg1 step %c1 iter_args(%arg5 = %c0_i16) -> (i16) {
        %extracted = tensor.extract %arg0[%arg4] : tensor<32xi16>
        %2 = arith.addi %extracted, %arg5 : i16
        scf.yield %2 : i16
    }
    // CHECK: return
    return %1 : i16
}
