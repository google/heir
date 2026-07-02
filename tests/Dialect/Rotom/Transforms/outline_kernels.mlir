// RUN: heir-opt %s --rotom-assign-layout --rotom-outline-kernels --mlir-print-local-scope | FileCheck %s

// Two matvecs with the same layout signature share one outlined kernel
// function (deduplicated by operand/result layouts and types); each call
// site carries the kernel's result layout. The kernel body keeps the
// rotom.matmul layout combination so the materializer and the ciphertext
// lowering process it like any other layout-assigned function.

#seed_mat = #rotom.seed<layouts = [#rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:4:1]>], n = 16>]>
#seed_col = #rotom.seed<layouts = [#rotom.layout<dims = [#rotom.dim<[0:4:1]>], n = 16>]>

module {
  // CHECK: func.func @two_matvecs
  // CHECK: call @rotom_kernel_matmul_0
  // CHECK-SAME: rotom.layout
  // CHECK: call @rotom_kernel_matmul_0
  // CHECK-NOT: linalg.matmul
  func.func @two_matvecs(%a: tensor<4x4xf32> {rotom.seed = #seed_mat}, %c: tensor<4x4xf32> {rotom.seed = #seed_mat}, %b: tensor<4x1xf32> {rotom.seed = #seed_col}) -> (tensor<4x1xf32>, tensor<4x1xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<4x1xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x1xf32>) -> tensor<4x1xf32>
    %0 = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x1xf32>) outs(%fill : tensor<4x1xf32>) -> tensor<4x1xf32>
    %1 = linalg.matmul ins(%c, %b : tensor<4x4xf32>, tensor<4x1xf32>) outs(%fill : tensor<4x1xf32>) -> tensor<4x1xf32>
    return %0, %1 : tensor<4x1xf32>, tensor<4x1xf32>
  }

  // Exactly one kernel function, marked and carrying the layout signature.
  // CHECK: func.func private @rotom_kernel_matmul_0
  // CHECK-SAME: rotom.layout
  // CHECK-SAME: rotom.kernel_func
  // CHECK: linalg.matmul
  // CHECK-SAME: rotom.matmul
  // CHECK-NOT: func.func private @rotom_kernel_matmul_1
}
