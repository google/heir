// RUN: heir-opt --layout-propagation=ciphertext-size=8 --mlir-print-local-scope --fold-convert-layout-into-assign-layout %s | FileCheck %s

module {
  // CHECK: func @matmul
  func.func @matmul(%arg0: !secret.secret<tensor<3x2xf32>>, %arg1: tensor<2x3xf32>) -> !secret.secret<tensor<3x3xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<3x3xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<3x2xf32>>) {
    ^body(%input0: tensor<3x2xf32>):
      // CHECK: linalg.matmul
      // CHECK-SAME: #secret.kernel<name = "MatmulDiagonal"
      %1 = linalg.matmul ins(%input0, %arg1 : tensor<3x2xf32>, tensor<2x3xf32>) outs(%cst : tensor<3x3xf32>) -> tensor<3x3xf32>
      secret.yield %1 : tensor<3x3xf32>
    } -> !secret.secret<tensor<3x3xf32>>
    return %0 : !secret.secret<tensor<3x3xf32>>
  }

  // CHECK: func @matmul_secret_secret
  func.func @matmul_secret_secret(%arg0: !secret.secret<tensor<3x5xf32>>, %arg1: !secret.secret<tensor<5x2xf32>>) -> !secret.secret<tensor<3x2xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<3x2xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<3x5xf32>>, %arg1: !secret.secret<tensor<5x2xf32>>) {
    ^body(%input0: tensor<3x5xf32>, %input1: tensor<5x2xf32>):
      // CHECK: linalg.matmul
      // CHECK-SAME: #secret.kernel<name = "MatmulBicyclic"
      %1 = linalg.matmul ins(%input0, %input1 : tensor<3x5xf32>, tensor<5x2xf32>) outs(%cst : tensor<3x2xf32>) -> tensor<3x2xf32>
      secret.yield %1 : tensor<3x2xf32>
    } -> !secret.secret<tensor<3x2xf32>>
    return %0 : !secret.secret<tensor<3x2xf32>>
  }
}
