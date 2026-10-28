// RUN: heir-opt %s --layout-propagation=min-slot-count=32 --convert-to-ciphertext-semantics=min-slot-count=32 | FileCheck %s

// A division by a packed cleartext becomes a product with its reciprocal,
// which is zero where the packed divisor is zero.

// CHECK: func.func @divide_by_cleartext
// CHECK: ^body(%[[x:[^:]*]]: tensor<
// CHECK: %[[isZero:.*]] = arith.cmpf oeq, %[[c:.*]], %[[zero:.*]] :
// CHECK: %[[reciprocal:.*]] = arith.divf %{{.*}}, %[[c]]
// CHECK: %[[factor:.*]] = arith.select %[[isZero]], %[[zero]], %[[reciprocal]]
// CHECK: arith.mulf %[[x]], %[[factor]]
// CHECK: return
func.func @divide_by_cleartext(%x: !secret.secret<tensor<8xf32>>, %c: tensor<8xf32>) -> !secret.secret<tensor<8xf32>> {
  %0 = secret.generic(%x: !secret.secret<tensor<8xf32>>) {
  ^body(%input: tensor<8xf32>):
    %1 = arith.divf %input, %c : tensor<8xf32>
    secret.yield %1 : tensor<8xf32>
  } -> !secret.secret<tensor<8xf32>>
  return %0 : !secret.secret<tensor<8xf32>>
}

// A division by a secret is left alone.

// CHECK: func.func @divide_by_secret
// CHECK: ^body(%[[x:[^:]*]]: tensor<{{.*}}>, %[[y:[^:]*]]: tensor<
// CHECK-NOT: arith.select
// CHECK: arith.divf %[[x]], %[[y]]
// CHECK: return
func.func @divide_by_secret(%x: !secret.secret<tensor<8xf32>>, %y: !secret.secret<tensor<8xf32>>) -> !secret.secret<tensor<8xf32>> {
  %0 = secret.generic(%x: !secret.secret<tensor<8xf32>>, %y: !secret.secret<tensor<8xf32>>) {
  ^body(%input0: tensor<8xf32>, %input1: tensor<8xf32>):
    %1 = arith.divf %input0, %input1 : tensor<8xf32>
    secret.yield %1 : tensor<8xf32>
  } -> !secret.secret<tensor<8xf32>>
  return %0 : !secret.secret<tensor<8xf32>>
}
