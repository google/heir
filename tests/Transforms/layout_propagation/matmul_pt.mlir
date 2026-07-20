// RUN: heir-opt --layout-propagation=ciphertext-size=64 --mlir-print-local-scope --fold-convert-layout-into-assign-layout %s | FileCheck %s

module {
  // CHECK: func @matmul_ctpt
  // CHECK: linalg.matmul
  // CHECK-SAME: #secret.kernel<name = "MatmulBicyclicDiagonal"
  func.func @matmul_ctpt(%arg0: !secret.secret<tensor<3x5xf32>>, %arg1: tensor<5x7xf32>) -> !secret.secret<tensor<3x7xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<3x7xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<3x5xf32>>) {
    ^body(%input0: tensor<3x5xf32>):
      %1 = linalg.matmul ins(%input0, %arg1 : tensor<3x5xf32>, tensor<5x7xf32>) outs(%cst : tensor<3x7xf32>) -> tensor<3x7xf32>
      secret.yield %1 : tensor<3x7xf32>
    } -> !secret.secret<tensor<3x7xf32>>
    return %0 : !secret.secret<tensor<3x7xf32>>
  }

  // CHECK: func @matmul_ptct
  // CHECK: linalg.matmul
  // CHECK-SAME: #secret.kernel<name = "MatmulBicyclicDiagonal"
  func.func @matmul_ptct(%arg0: tensor<3x5xf32>, %arg1: !secret.secret<tensor<5x7xf32>>) -> !secret.secret<tensor<3x7xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<3x7xf32>
    %0 = secret.generic(%arg1: !secret.secret<tensor<5x7xf32>>) {
    ^body(%input1: tensor<5x7xf32>):
      %1 = linalg.matmul ins(%arg0, %input1 : tensor<3x5xf32>, tensor<5x7xf32>) outs(%cst : tensor<3x7xf32>) -> tensor<3x7xf32>
      secret.yield %1 : tensor<3x7xf32>
    } -> !secret.secret<tensor<3x7xf32>>
    return %0 : !secret.secret<tensor<3x7xf32>>
  }

  // CHECK: func @matmul_non_coprime
  // CHECK: linalg.matmul
  // CHECK-SAME: #secret.kernel<name = "MatmulDiagonal"
  func.func @matmul_non_coprime(%arg0: !secret.secret<tensor<4x6xf32>>, %arg1: tensor<6x9xf32>) -> !secret.secret<tensor<4x9xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<4x9xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<4x6xf32>>) {
    ^body(%input0: tensor<4x6xf32>):
      %1 = linalg.matmul ins(%input0, %arg1 : tensor<4x6xf32>, tensor<6x9xf32>) outs(%cst : tensor<4x9xf32>) -> tensor<4x9xf32>
      secret.yield %1 : tensor<4x9xf32>
    } -> !secret.secret<tensor<4x9xf32>>
    return %0 : !secret.secret<tensor<4x9xf32>>
  }

  // CHECK: func @matmul_non_coprime_output
  // CHECK: linalg.matmul
  // CHECK-SAME: #secret.kernel<name = "MatmulDiagonal"
  func.func @matmul_non_coprime_output(%arg0: !secret.secret<tensor<3x5xf32>>, %arg1: tensor<5x9xf32>) -> !secret.secret<tensor<3x9xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<3x9xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<3x5xf32>>) {
    ^body(%input0: tensor<3x5xf32>):
      %1 = linalg.matmul ins(%input0, %arg1 : tensor<3x5xf32>, tensor<5x9xf32>) outs(%cst : tensor<3x9xf32>) -> tensor<3x9xf32>
      secret.yield %1 : tensor<3x9xf32>
    } -> !secret.secret<tensor<3x9xf32>>
    return %0 : !secret.secret<tensor<3x9xf32>>
  }

  // CHECK: func @matmul_trivial_dim
  // CHECK: linalg.matmul
  // CHECK-SAME: #secret.kernel<name = "MatmulBicyclicDiagonal"
  func.func @matmul_trivial_dim(%arg0: !secret.secret<tensor<1x4xf32>>, %arg1: tensor<4x7xf32>) -> !secret.secret<tensor<1x7xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x7xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<1x4xf32>>) {
    ^body(%input0: tensor<1x4xf32>):
      %1 = linalg.matmul ins(%input0, %arg1 : tensor<1x4xf32>, tensor<4x7xf32>) outs(%cst : tensor<1x7xf32>) -> tensor<1x7xf32>
      secret.yield %1 : tensor<1x7xf32>
    } -> !secret.secret<tensor<1x7xf32>>
    return %0 : !secret.secret<tensor<1x7xf32>>
  }
}
