// RUN: heir-opt %s --rotom-seed-layout=n=16 --rotom-assign-layout --rotom-materialize-tensor-ext-layout --mlir-print-local-scope | FileCheck %s

module {
  // CHECK: func.func @chained_matmul(
  // CHECK-SAME: tensor_ext.layout = #tensor_ext.layout
  // CHECK-SAME: tensor_ext.layout = #tensor_ext.layout
  // CHECK-SAME: tensor_ext.layout = #tensor_ext.layout
  func.func @chained_matmul(%lhs: !secret.secret<tensor<4x4xf32>>, %mid: tensor<4x4xf32>, %rhs: tensor<4x4xf32>) -> !secret.secret<tensor<4x4xf32>> {
    %0 = secret.generic(%lhs : !secret.secret<tensor<4x4xf32>>) {
    ^bb0(%lhs_plain: tensor<4x4xf32>):
      %empty0 = tensor.empty() : tensor<4x4xf32>
      // CHECK: %[[MM0:.*]] = linalg.matmul
      // CHECK-SAME: tensor_ext.layout = #tensor_ext.layout
      // CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<4x4xf32>, tensor<4x4xf32>)
      %mm0 = linalg.matmul ins(%lhs_plain, %mid : tensor<4x4xf32>, tensor<4x4xf32>) outs(%empty0 : tensor<4x4xf32>) -> tensor<4x4xf32>
      %empty1 = tensor.empty() : tensor<4x4xf32>
      // CHECK: %[[MM1:.*]] = linalg.matmul
      // CHECK-SAME: tensor_ext.layout = #tensor_ext.layout
      // CHECK-SAME: ins(%[[MM0]], %{{.*}} : tensor<4x4xf32>, tensor<4x4xf32>)
      %mm1 = linalg.matmul ins(%mm0, %rhs : tensor<4x4xf32>, tensor<4x4xf32>) outs(%empty1 : tensor<4x4xf32>) -> tensor<4x4xf32>
      // CHECK: secret.yield %[[MM1]]
      secret.yield %mm1 : tensor<4x4xf32>
    } -> !secret.secret<tensor<4x4xf32>>
    return %0 : !secret.secret<tensor<4x4xf32>>
  }
}
