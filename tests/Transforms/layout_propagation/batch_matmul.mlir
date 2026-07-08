// RUN: heir-opt --layout-propagation=ciphertext-size=8 --mlir-print-local-scope --fold-convert-layout-into-assign-layout %s | FileCheck %s

module {
  // CHECK: func @batch_matmul_secret_secret
  func.func @batch_matmul_secret_secret(%arg0: !secret.secret<tensor<2x3x5xf32>>, %arg1: !secret.secret<tensor<2x5x7xf32>>) -> !secret.secret<tensor<2x3x7xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<2x3x7xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<2x3x5xf32>>, %arg1: !secret.secret<tensor<2x5x7xf32>>) {
    ^body(%input0: tensor<2x3x5xf32>, %input1: tensor<2x5x7xf32>):
      // CHECK: linalg.batch_matmul
      // CHECK-SAME: #secret.kernel<name = "BatchMatmulTricyclic"
      %1 = linalg.batch_matmul ins(%input0, %input1 : tensor<2x3x5xf32>, tensor<2x5x7xf32>) outs(%cst : tensor<2x3x7xf32>) -> tensor<2x3x7xf32>
      secret.yield %1 : tensor<2x3x7xf32>
    } -> !secret.secret<tensor<2x3x7xf32>>
    return %0 : !secret.secret<tensor<2x3x7xf32>>
  }
}
