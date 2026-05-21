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

#layout_replicated_dim = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 4>, #rotom.dim<dim = -1, size = 2, stride = 4>], n = 8>
#seed_replicated_dim = #rotom.seed<layouts = [#layout_replicated_dim]>

module {
  // CHECK: func.func @rotom_add_replicated_dim
  // CHECK-SAME: tensor<1x16xf32>
  // CHECK-NOT: arith.addf %arg0, %arg1
  // CHECK: tensor_ext.remap
  // CHECK: arith.addf
  func.func @rotom_add_replicated_dim(%arg0: tensor<4xf32> {rotom.seed = #seed_replicated_dim}, %arg1: tensor<4xf32> {rotom.seed = #seed_replicated_dim}) -> tensor<4xf32> {
    %0 = arith.addf %arg0, %arg1 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// -----

#layout_rolled_tiled = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 2, stride = 2>, #rotom.dim<dim = 1, size = 2, stride = 2>, #rotom.dim<dim = 0, size = 2>, #rotom.dim<dim = 1, size = 2>], n = 8>
#seed_rolled_tiled = #rotom.seed<layouts = [#layout_rolled_tiled]>

module {
  // CHECK: func.func @rolled_tiled_add_not_rotom_lowered
  // CHECK-SAME: tensor<2x2x2x2xf32>
  // CHECK-NOT: tensor_ext.remap
  // CHECK: arith.addf
  func.func @rolled_tiled_add_not_rotom_lowered(%arg0: tensor<2x2x2x2xf32> {rotom.seed = #seed_rolled_tiled}, %arg1: tensor<2x2x2x2xf32> {rotom.seed = #seed_rolled_tiled}) -> tensor<2x2x2x2xf32> {
    %0 = arith.addf %arg0, %arg1 : tensor<2x2x2x2xf32>
    return %0 : tensor<2x2x2x2xf32>
  }
}
