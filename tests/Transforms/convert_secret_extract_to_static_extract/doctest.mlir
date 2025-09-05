// RUN: heir-opt --convert-secret-extract-to-static-extract %s | FileCheck %s

// CHECK: func.func @main
// CHECK:      %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[C32:.*]] = arith.constant 32 : index
// CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
// CHECK:      %[[GENERIC:.*]] = secret.generic
// CHECK:      ^bb0(%[[TENSOR:.*]]: tensor<32xi16>, %[[INDEX:.*]]: index):
// CHECK-NEXT:   %[[C0_I16:.*]] = arith.constant 0 : i16
// CHECK-NEXT:   %[[FOR:.*]] = affine.for %[[IV:.*]] = 0 to 32 iter_args(%[[ARG:.*]] = %[[C0_I16]]) -> (i16) {
// CHECK-NEXT:     %[[CMP:.*]] = arith.cmpi eq, %[[IV]], %[[INDEX]] : index
// CHECK-NEXT:     %[[EXTRACT:.*]] = tensor.extract %[[TENSOR]][%[[IV]]] : tensor<32xi16>
// CHECK-NEXT:     %[[IF:.*]] = scf.if %[[CMP]] -> (i16) {
// CHECK-NEXT:       scf.yield %[[EXTRACT]] : i16
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %[[ARG]] : i16
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.yield %[[IF]] : i16
// CHECK-NEXT:   }
// CHECK-NEXT:   secret.yield %[[FOR]] : i16
// CHECK:      return

func.func @main(%secretTensor: !secret.secret<tensor<32xi16>>, %secretIndex: !secret.secret<index>) -> !secret.secret<i16> {
  %0 = secret.generic(%secretTensor, %secretIndex : !secret.secret<tensor<32xi16>>, !secret.secret<index>) {
  ^bb0(%tensor: tensor<32xi16>, %index: index):
    // Violation: tensor.extract loads value at secret index
    %extractedValue = tensor.extract %tensor[%index] : tensor<32xi16>
    secret.yield %extractedValue : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}
