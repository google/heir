// RUN: heir-opt --convert-secret-insert-to-static-insert --split-input-file --verify-diagnostics %s

// CHECK-LABEL: @insert_without_size_attr
func.func @insert_without_size_attr(%arg0: !secret.secret<tensor<16xi32>>, %arg1: !secret.secret<index>) -> !secret.secret<tensor<16xi32>> {
  %c10_i32 = arith.constant 10 : i32
  %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<tensor<16xi32>>, !secret.secret<index>) {
  ^bb0(%arg2: tensor<16xi32>, %arg3: index):
    // expected-warning@+1 {{Cannot convert secret tensor.insert to static tensor.insert since a size attribute (`size`) has not been provided on the tensor.insert op:}}
    %inserted = tensor.insert %c10_i32 into %arg2[%arg3] : tensor<16xi32>
    secret.yield %inserted : tensor<16xi32>
  } -> !secret.secret<tensor<16xi32>>
  return %0 : !secret.secret<tensor<16xi32>>
}

// -----
