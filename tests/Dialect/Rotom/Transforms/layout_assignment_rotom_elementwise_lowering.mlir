// RUN: heir-opt %s --rotom-assign-layout --rotom-materialize-tensor-ext-layout --convert-to-ciphertext-semantics="ciphertext-size=16" --mlir-print-local-scope | FileCheck %s

#layout_a = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 4>, #rotom.dim<dim = 1, size = 4>], n = 16>
#layout_b = #rotom.layout<dims = [#rotom.dim<dim = 1, size = 4>, #rotom.dim<dim = 0, size = 4>], n = 16>
#seed_a = #rotom.seed<layouts = [#layout_a]>
#seed_b = #rotom.seed<layouts = [#layout_b]>

module {
  // CHECK: func.func @rotom_add
  // CHECK-SAME: tensor<1x16xf32>
  // CHECK-NOT: arith.addf %arg0, %arg1
  // CHECK: tensor_ext.remap
  // CHECK: arith.addf
  func.func @rotom_add(%arg0: tensor<4x4xf32> {rotom.seed = #seed_a}, %arg1: tensor<4x4xf32> {rotom.seed = #seed_b}) -> tensor<4x4xf32> {
    %0 = arith.addf %arg0, %arg1 : tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  // CHECK: func.func @rotom_mul
  // CHECK-SAME: tensor<1x16xf32>
  // CHECK-NOT: arith.mulf %arg0, %arg1
  // CHECK: tensor_ext.remap
  // CHECK: arith.mulf
  // CHECK: arith.addf
  func.func @rotom_mul(%arg0: tensor<4x4xf32> {rotom.seed = #seed_a}, %arg1: tensor<4x4xf32> {rotom.seed = #seed_b}) -> tensor<4x4xf32> {
    %0 = arith.mulf %arg0, %arg1 : tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
