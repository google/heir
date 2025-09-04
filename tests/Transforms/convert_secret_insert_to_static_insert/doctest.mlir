// RUN: heir-opt --convert-secret-insert-to-static-insert %s | FileCheck %s

// CHECK: func.func @main
// CHECK:      %[[C0_I16:.*]] = arith.constant 0 : i16
// CHECK:      %[[GENERIC:.*]] = secret.generic
// CHECK:      ^body(%[[TENSOR:.*]]: tensor<32xi16>, %[[INDEX:.*]]: index):
// CHECK-NEXT:   %[[FOR:.*]] = affine.for %[[IV:.*]] = 0 to 32 iter_args(%[[ARG:.*]] = %[[TENSOR]]) -> (tensor<32xi16>) {
// CHECK-NEXT:     %[[CMP:.*]] = arith.cmpi eq, %[[IV]], %[[INDEX]] : index
// CHECK-NEXT:     %[[INSERT:.*]] = tensor.insert %[[C0_I16]] into %[[ARG]][%[[IV]]] : tensor<32xi16>
// CHECK-NEXT:     %[[IF:.*]] = scf.if %[[CMP]] -> (tensor<32xi16>) {
// CHECK-NEXT:       scf.yield %[[INSERT]] : tensor<32xi16>
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %[[ARG]] : tensor<32xi16>
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.yield %[[IF]] : tensor<32xi16>
// CHECK-NEXT:   }
// CHECK-NEXT:   secret.yield %[[FOR]] : tensor<32xi16>
// CHECK:      return

func.func @main(%secretTensor: !secret.secret<tensor<32xi16>>, %secretIndex: !secret.secret<index>) -> !secret.secret<tensor<32xi16>> {
  %c0 = arith.constant 0 : i16
  %0 = secret.generic(%secretTensor: !secret.secret<tensor<32xi16>>, %secretIndex: !secret.secret<index>) {
  ^bb0(%tensor: tensor<32xi16>, %index: index):
    // Violation: tensor.insert writes value at secret index
    %inserted = tensor.insert %c0 into %tensor[%index] : tensor<32xi16>
    secret.yield %inserted : tensor<32xi16>
  } -> !secret.secret<tensor<32xi16>>
  return %0 : !secret.secret<tensor<32xi16>>
}
