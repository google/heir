// RUN: heir-opt %s --rotom-assign-layout --rotom-materialize-tensor-ext-layout --convert-to-ciphertext-semantics="ciphertext-size=16" --mlir-print-local-scope | FileCheck %s

// Elementwise ops over func-arg operands. The two seeds are row-major and
// column-major packings of the same tensor -- genuinely different layouts --
// so the lowering aligns the second operand onto the compute layout with a
// single tensor_ext.remap (the transpose permutation) and then emits one bare
// elementwise op. (The replicated-dim case below seeds both operands
// identically, so no remap is needed there.)

#layout_a = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:4:1]>], n = 16>
#layout_b = #rotom.layout<dims = [#rotom.dim<[1:4:1]>, #rotom.dim<[0:4:1]>], n = 16>
#seed_a = #rotom.seed<layouts = [#layout_a]>
#seed_b = #rotom.seed<layouts = [#layout_b]>

module {
  // CHECK: func.func @rotom_add
  // CHECK-SAME: tensor<1x16xf32>
  // CHECK: %[[ALIGNED_ADDF:.*]] = tensor_ext.remap %arg1
  // CHECK: arith.addf %arg0, %[[ALIGNED_ADDF]]
  func.func @rotom_add(%arg0: tensor<4x4xf32> {rotom.seed = #seed_a}, %arg1: tensor<4x4xf32> {rotom.seed = #seed_b}) -> tensor<4x4xf32> {
    %0 = arith.addf %arg0, %arg1 : tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  // CHECK: func.func @rotom_mul
  // CHECK-SAME: tensor<1x16xf32>
  // CHECK: %[[ALIGNED_MULF:.*]] = tensor_ext.remap %arg1
  // CHECK: arith.mulf %arg0, %[[ALIGNED_MULF]]
  func.func @rotom_mul(%arg0: tensor<4x4xf32> {rotom.seed = #seed_a}, %arg1: tensor<4x4xf32> {rotom.seed = #seed_b}) -> tensor<4x4xf32> {
    %0 = arith.mulf %arg0, %arg1 : tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  // CHECK: func.func @rotom_addi
  // CHECK-SAME: tensor<1x16xi16>
  // CHECK: %[[ALIGNED_ADDI:.*]] = tensor_ext.remap %arg1
  // CHECK: arith.addi %arg0, %[[ALIGNED_ADDI]]
  func.func @rotom_addi(%arg0: tensor<4x4xi16> {rotom.seed = #seed_a}, %arg1: tensor<4x4xi16> {rotom.seed = #seed_b}) -> tensor<4x4xi16> {
    %0 = arith.addi %arg0, %arg1 : tensor<4x4xi16>
    return %0 : tensor<4x4xi16>
  }

  // CHECK: func.func @rotom_muli
  // CHECK-SAME: tensor<1x16xi16>
  // CHECK: %[[ALIGNED_MULI:.*]] = tensor_ext.remap %arg1
  // CHECK: arith.muli %arg0, %[[ALIGNED_MULI]]
  func.func @rotom_muli(%arg0: tensor<4x4xi16> {rotom.seed = #seed_a}, %arg1: tensor<4x4xi16> {rotom.seed = #seed_b}) -> tensor<4x4xi16> {
    %0 = arith.muli %arg0, %arg1 : tensor<4x4xi16>
    return %0 : tensor<4x4xi16>
  }

  // Subtraction is additive: it shares the RotomAdd kernel and lowers like add.
  // CHECK: func.func @rotom_sub
  // CHECK-SAME: tensor<1x16xf32>
  // CHECK: %[[ALIGNED_SUBF:.*]] = tensor_ext.remap %arg1
  // CHECK: arith.subf %arg0, %[[ALIGNED_SUBF]]
  func.func @rotom_sub(%arg0: tensor<4x4xf32> {rotom.seed = #seed_a}, %arg1: tensor<4x4xf32> {rotom.seed = #seed_b}) -> tensor<4x4xf32> {
    %0 = arith.subf %arg0, %arg1 : tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  // CHECK: func.func @rotom_subi
  // CHECK-SAME: tensor<1x16xi16>
  // CHECK: %[[ALIGNED_SUBI:.*]] = tensor_ext.remap %arg1
  // CHECK: arith.subi %arg0, %[[ALIGNED_SUBI]]
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
  // A tiled (non-unit-stride) layout: it flows through assignment and
  // materialization (the 4x4 packs into 2 ciphertexts), but is not Rotom-lowered
  // -- the add stays a plain ciphertext add with no conversion.
  // CHECK: func.func @rolled_tiled_add_not_rotom_lowered
  // CHECK-SAME: tensor<2x16xf32>
  // CHECK-NOT: tensor_ext.remap
  // CHECK: arith.addf
  func.func @rolled_tiled_add_not_rotom_lowered(%arg0: tensor<4x4xf32> {rotom.seed = #seed_rolled_tiled}, %arg1: tensor<4x4xf32> {rotom.seed = #seed_rolled_tiled}) -> tensor<4x4xf32> {
    %0 = arith.addf %arg0, %arg1 : tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
