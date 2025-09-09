// RUN: heir-opt --convert-secret-for-to-static-for %s | FileCheck %s

// CHECK: func.func @main
// CHECK:      %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:      %[[GENERIC:.*]] = secret.generic
// CHECK:      ^body(%[[TENSOR:.*]]: tensor<16xi32>, %[[LOWER:.*]]: index, %[[UPPER:.*]]: index):
// CHECK-NEXT:   %[[FOR:.*]] = affine.for %[[IV:.*]] = 0 to 16 iter_args(%[[ARG:.*]] = %[[C0_I32]]) -> (i32) {
// CHECK-NEXT:     %[[CMP_LOWER:.*]] = arith.cmpi sge, %[[IV]], %[[LOWER]] : index
// CHECK-NEXT:     %[[CMP_UPPER:.*]] = arith.cmpi slt, %[[IV]], %[[UPPER]] : index
// CHECK-NEXT:     %[[AND:.*]] = arith.andi %[[CMP_LOWER]], %[[CMP_UPPER]] : i1
// CHECK-NEXT:     %[[IF:.*]] = scf.if %[[AND]] -> (i32) {
// CHECK-NEXT:       %[[EXTRACT:.*]] = tensor.extract %[[TENSOR]][%[[IV]]] : tensor<16xi32>
// CHECK-NEXT:       %[[ADD:.*]] = arith.addi %[[EXTRACT]], %[[ARG]] : i32
// CHECK-NEXT:       scf.yield %[[ADD]] : i32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %[[ARG]] : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.yield %[[IF]] : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   secret.yield %[[FOR]] : i32
// CHECK:      return

func.func @main(%secretTensor: !secret.secret<tensor<16xi32>>, %secretLower: !secret.secret<index>, %secretUpper: !secret.secret<index>) -> !secret.secret<i32> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : index
  %0 = secret.generic(%secretTensor: !secret.secret<tensor<16xi32>>, %secretLower: !secret.secret<index>, %secretUpper: !secret.secret<index>) {
  ^bb0(%tensor: tensor<16xi32>, %lower : index, %upper : index ):
    %1 = scf.for %i = %lower to %upper step %c1 iter_args(%arg = %c0) -> (i32) {
      %extracted = tensor.extract %tensor[%i] : tensor<16xi32>
      %sum = arith.addi %extracted, %arg : i32
      scf.yield %sum : i32
    } {lower = 0, upper = 16}
    secret.yield %1 : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}
