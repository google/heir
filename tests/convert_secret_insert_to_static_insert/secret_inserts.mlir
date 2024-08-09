// RUN: heir-opt --convert-secret-insert-to-static-insert %s | FileCheck %s

// CHECK-LABEL: @insert_to_secret_index
func.func @insert_to_secret_index(%arg0: !secret.secret<tensor<16xi32>>, %arg1: !secret.secret<index>) -> !secret.secret<tensor<16xi32>> {
  %c10_i32 = arith.constant 10 : i32
  %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<tensor<16xi32>>, !secret.secret<index>) {
  ^bb0(%arg2: tensor<16xi32>, %arg3: index):
    // CHECK:      %[[INSERTED:.*]] = tensor.insert
    // CHECK-NEXT: %[[IF:.*]] = scf.if %[[COND:.*]] -> (tensor<16xi32>) {
    // CHECK-NEXT:   scf.yield %[[INSERTED]] : tensor<16xi32>
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   scf.yield %[[OLD_TENSOR:.*]] : tensor<16xi32>
    // CHECK-NEXT: }
    %inserted = tensor.insert %c10_i32 into %arg2[%arg3] {size = 16} : tensor<16xi32>
    secret.yield %inserted : tensor<16xi32>
  } -> !secret.secret<tensor<16xi32>>
  return %0 : !secret.secret<tensor<16xi32>>
}
