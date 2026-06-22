// RUN: heir-opt %s --rotom-assign-layout --rotom-materialize-tensor-ext-layout --convert-to-ciphertext-semantics="ciphertext-size=16" --mlir-print-local-scope | FileCheck %s

// Elementwise ops over func-arg operands: the materializer unifies the
// operands onto a single (consumer) layout at the function boundary, so the
// lowering is a bare elementwise op with no layout conversion. (A conversion
// is only emitted for *internal* values whose producer pins a fixed layout;
// see the pipeline execution test for the conversion path.)

#layout_a = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:4:1]>], n = 16>
#layout_b = #rotom.layout<dims = [#rotom.dim<[1:4:1]>, #rotom.dim<[0:4:1]>], n = 16>
#seed_a = #rotom.seed<layouts = [#layout_a]>
#seed_b = #rotom.seed<layouts = [#layout_b]>

module {
  // CHECK: func.func @rotom_add
  // CHECK-SAME: tensor<1x16xf32>
  // CHECK-NOT: tensor_ext.remap
  // CHECK: arith.addf %arg0, %arg1
  func.func @rotom_add(%arg0: tensor<4x4xf32> {rotom.seed = #seed_a}, %arg1: tensor<4x4xf32> {rotom.seed = #seed_b}) -> tensor<4x4xf32> {
    %0 = arith.addf %arg0, %arg1 : tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  // CHECK: func.func @rotom_mul
  // CHECK-SAME: tensor<1x16xf32>
  // CHECK-NOT: tensor_ext.remap
  // CHECK: arith.mulf %arg0, %arg1
  func.func @rotom_mul(%arg0: tensor<4x4xf32> {rotom.seed = #seed_a}, %arg1: tensor<4x4xf32> {rotom.seed = #seed_b}) -> tensor<4x4xf32> {
    %0 = arith.mulf %arg0, %arg1 : tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  // CHECK: func.func @rotom_addi
  // CHECK-SAME: tensor<1x16xi16>
  // CHECK-NOT: tensor_ext.remap
  // CHECK: arith.addi %arg0, %arg1
  func.func @rotom_addi(%arg0: tensor<4x4xi16> {rotom.seed = #seed_a}, %arg1: tensor<4x4xi16> {rotom.seed = #seed_b}) -> tensor<4x4xi16> {
    %0 = arith.addi %arg0, %arg1 : tensor<4x4xi16>
    return %0 : tensor<4x4xi16>
  }

  // CHECK: func.func @rotom_muli
  // CHECK-SAME: tensor<1x16xi16>
  // CHECK-NOT: tensor_ext.remap
  // CHECK: arith.muli %arg0, %arg1
  func.func @rotom_muli(%arg0: tensor<4x4xi16> {rotom.seed = #seed_a}, %arg1: tensor<4x4xi16> {rotom.seed = #seed_b}) -> tensor<4x4xi16> {
    %0 = arith.muli %arg0, %arg1 : tensor<4x4xi16>
    return %0 : tensor<4x4xi16>
  }

  // Subtraction is additive: it shares the RotomAdd kernel and lowers like add.
  // CHECK: func.func @rotom_sub
  // CHECK-SAME: tensor<1x16xf32>
  // CHECK-NOT: tensor_ext.remap
  // CHECK: arith.subf %arg0, %arg1
  func.func @rotom_sub(%arg0: tensor<4x4xf32> {rotom.seed = #seed_a}, %arg1: tensor<4x4xf32> {rotom.seed = #seed_b}) -> tensor<4x4xf32> {
    %0 = arith.subf %arg0, %arg1 : tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  // CHECK: func.func @rotom_subi
  // CHECK-SAME: tensor<1x16xi16>
  // CHECK-NOT: tensor_ext.remap
  // CHECK: arith.subi %arg0, %arg1
  func.func @rotom_subi(%arg0: tensor<4x4xi16> {rotom.seed = #seed_a}, %arg1: tensor<4x4xi16> {rotom.seed = #seed_b}) -> tensor<4x4xi16> {
    %0 = arith.subi %arg0, %arg1 : tensor<4x4xi16>
    return %0 : tensor<4x4xi16>
  }
}

#layout_replicated_dim = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[-1:2:4]>], n = 8>
#seed_replicated_dim = #rotom.seed<layouts = [#layout_replicated_dim]>

module {
  // CHECK: func.func @rotom_add_replicated_dim
  // CHECK-SAME: tensor<1x16xf32>
  // CHECK-NOT: tensor_ext.remap
  // CHECK: arith.addf %arg0, %arg1
  func.func @rotom_add_replicated_dim(%arg0: tensor<4xf32> {rotom.seed = #seed_replicated_dim}, %arg1: tensor<4xf32> {rotom.seed = #seed_replicated_dim}) -> tensor<4xf32> {
    %0 = arith.addf %arg0, %arg1 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// -----

#layout_rolled_tiled = #rotom.layout<dims = [#rotom.dim<[0:2:2]>, #rotom.dim<[1:2:2]>, #rotom.dim<[0:2:1]>, #rotom.dim<[1:2:1]>], n = 8>
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
