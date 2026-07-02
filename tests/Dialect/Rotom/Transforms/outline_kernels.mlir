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

  // Elementwise Rotom kernels outline and deduplicate the same way: the two
  // addf ops (one over an intermediate with the same layout) share a kernel,
  // and the mulf gets its own.
  // CHECK: func.func @elementwise_chain
  // CHECK: %[[S:.*]] = call @rotom_kernel_addf_1(%arg0, %arg1)
  // CHECK: %[[T:.*]] = call @rotom_kernel_addf_1(%[[S]], %arg2)
  // CHECK: call @rotom_kernel_mulf_2(%[[T]], %arg0)
  // CHECK-NOT: arith.addf
  func.func @elementwise_chain(%a: tensor<4x4xf32> {rotom.seed = #seed_mat}, %b: tensor<4x4xf32> {rotom.seed = #seed_mat}, %c: tensor<4x4xf32> {rotom.seed = #seed_mat}) -> tensor<4x4xf32> {
    %0 = arith.addf %a, %b : tensor<4x4xf32>
    %1 = arith.addf %0, %c : tensor<4x4xf32>
    %2 = arith.mulf %1, %a : tensor<4x4xf32>
    return %2 : tensor<4x4xf32>
  }

  // Exactly one kernel function per signature, marked and carrying it.
  // CHECK: func.func private @rotom_kernel_matmul_0
  // CHECK-SAME: rotom.layout
  // CHECK-SAME: rotom.kernel_func
  // CHECK: linalg.matmul
  // CHECK-SAME: rotom.matmul
  // CHECK: func.func private @rotom_kernel_addf_1
  // CHECK: arith.addf
  // CHECK-SAME: secret.kernel
  // CHECK: func.func private @rotom_kernel_mulf_2
  // CHECK-NOT: func.func private @rotom_kernel_addf
}
