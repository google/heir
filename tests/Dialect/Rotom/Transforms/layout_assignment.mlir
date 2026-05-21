// RUN: heir-opt %s --rotom-seed-layout=n=8 --rotom-assign-layout --mlir-print-local-scope | FileCheck %s

module {
  // CHECK: func.func @assign_from_seed(
  // CHECK-SAME: rotom.layout = #rotom.layout
  // CHECK-SAME: rotom.seed
  // CHECK-SAME: rotom.layout = #rotom.layout
  // CHECK-SAME: rotom.seed
  // CHECK-SAME: -> (!secret.secret<tensor<4x4xf32>> {rotom.layout = #rotom.layout
  func.func @assign_from_seed(%arg0: !secret.secret<tensor<4x4xf32>>, %arg1: tensor<4x4xf32>) -> !secret.secret<tensor<4x4xf32>> {
    // CHECK: %[[GENERIC:.*]] = secret.generic(
    // CHECK-SAME: rotom.layout = #rotom.layout
    %0 = secret.generic(%arg0 : !secret.secret<tensor<4x4xf32>>) {
    ^bb0(%arg2: tensor<4x4xf32>):
      // CHECK: arith.addf
      // CHECK-SAME: rotom.layout = #rotom.layout
      %1 = arith.addf %arg2, %arg1 : tensor<4x4xf32>
      secret.yield %1 : tensor<4x4xf32>
    } -> !secret.secret<tensor<4x4xf32>>
    // CHECK: return %[[GENERIC]]
    return %0 : !secret.secret<tensor<4x4xf32>>
  }
}

// -----

#layout_2x4 = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 2>, #rotom.dim<dim = 1, size = 4>], n = 8>
#seed_2x4 = #rotom.seed<layouts = [#layout_2x4]>

module {
  // CHECK: func.func @transpose_assign
  // CHECK-SAME: tensor<4x2xf32> {rotom.layout = #rotom.layout<dims = [#rotom.dim<dim = 1, size = 2>, #rotom.dim<dim = 0, size = 4>], n = 8>}
  func.func @transpose_assign(%arg0: tensor<2x4xf32> {rotom.seed = #seed_2x4}) -> tensor<4x2xf32> {
    %empty = tensor.empty() : tensor<4x2xf32>
    // CHECK: linalg.transpose
    // CHECK-SAME: rotom.layout = #rotom.layout<dims = [#rotom.dim<dim = 1, size = 2>, #rotom.dim<dim = 0, size = 4>], n = 8>
    %0 = linalg.transpose ins(%arg0 : tensor<2x4xf32>) outs(%empty : tensor<4x2xf32>) permutation = [1, 0]
    return %0 : tensor<4x2xf32>
  }
}

// -----

#layout_1x4 = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 1>, #rotom.dim<dim = 1, size = 4>], n = 8>
#layout_4 = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 4>], n = 8>
#seed_1x4 = #rotom.seed<layouts = [#layout_1x4]>
#seed_4 = #rotom.seed<layouts = [#layout_4]>

module {
  // CHECK: func.func @collapse_assign
  // CHECK-SAME: tensor<4xf32> {rotom.layout = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 4>], n = 8>}
  func.func @collapse_assign(%arg0: tensor<1x4xf32> {rotom.seed = #seed_1x4}) -> tensor<4xf32> {
    // CHECK: tensor.collapse_shape
    // CHECK-SAME: rotom.layout = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 4>], n = 8>
    %0 = tensor.collapse_shape %arg0 [[0, 1]] : tensor<1x4xf32> into tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CHECK: func.func @expand_assign
  // CHECK-SAME: tensor<1x4xf32> {rotom.layout = #rotom.layout<dims = [#rotom.dim<dim = 1, size = 4>], n = 8>}
  func.func @expand_assign(%arg0: tensor<4xf32> {rotom.seed = #seed_4}) -> tensor<1x4xf32> {
    // CHECK: tensor.expand_shape
    // CHECK-SAME: rotom.layout = #rotom.layout<dims = [#rotom.dim<dim = 1, size = 4>], n = 8>
    %0 = tensor.expand_shape %arg0 [[0, 1]] output_shape [1, 4] : tensor<4xf32> into tensor<1x4xf32>
    return %0 : tensor<1x4xf32>
  }
}

// -----

#layout_4d_slice = #rotom.layout<dims = [#rotom.dim<dim = 2, size = 4>, #rotom.dim<dim = 3, size = 4>], n = 16>
#layout_2d_slice = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 4>, #rotom.dim<dim = 1, size = 4>], n = 16>
#seed_4d_slice = #rotom.seed<layouts = [#layout_4d_slice]>
#seed_2d_slice = #rotom.seed<layouts = [#layout_2d_slice]>

module {
  // CHECK: func.func @extract_assign
  // CHECK-SAME: tensor<4x4xf32> {rotom.layout = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 4>, #rotom.dim<dim = 1, size = 4>], n = 16>}
  func.func @extract_assign(%arg0: tensor<1x2x4x4xf32> {rotom.seed = #seed_4d_slice}) -> tensor<4x4xf32> {
    // CHECK: tensor.extract_slice
    // CHECK-SAME: rotom.layout = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 4>, #rotom.dim<dim = 1, size = 4>], n = 16>
    %0 = tensor.extract_slice %arg0[0, 1, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] : tensor<1x2x4x4xf32> to tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  // CHECK: func.func @insert_assign
  // CHECK-SAME: tensor<1x2x4x4xf32> {rotom.layout = #rotom.layout<dims = [#rotom.dim<dim = 2, size = 4>, #rotom.dim<dim = 3, size = 4>], n = 16>}
  func.func @insert_assign(%arg0: tensor<4x4xf32> {rotom.seed = #seed_2d_slice}) -> tensor<1x2x4x4xf32> {
    %empty = tensor.empty() : tensor<1x2x4x4xf32>
    // CHECK: tensor.insert_slice
    // CHECK-SAME: rotom.layout = #rotom.layout<dims = [#rotom.dim<dim = 2, size = 4>, #rotom.dim<dim = 3, size = 4>], n = 16>
    %0 = tensor.insert_slice %arg0 into %empty[0, 1, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] : tensor<4x4xf32> into tensor<1x2x4x4xf32>
    return %0 : tensor<1x2x4x4xf32>
  }
}

// -----

#layout_reduce = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 2>, #rotom.dim<dim = 1, size = 4>], n = 8>
#seed_reduce = #rotom.seed<layouts = [#layout_reduce]>

module {
  // CHECK: func.func @reduce_assign
  // CHECK-SAME: tensor<4xf32> {rotom.layout = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 4>], n = 8>}
  func.func @reduce_assign(%arg0: tensor<2x4xf32> {rotom.seed = #seed_reduce}) -> tensor<4xf32> {
    %empty = tensor.empty() : tensor<4xf32>
    // CHECK: linalg.reduce
    // CHECK-SAME: rotom.layout = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 4>], n = 8>
    %0 = linalg.reduce ins(%arg0 : tensor<2x4xf32>) outs(%empty : tensor<4xf32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %1 = arith.addf %in, %init : f32
      linalg.yield %1 : f32
    }
    return %0 : tensor<4xf32>
  }
}

// -----

#layout_generic_a = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 2>, #rotom.dim<dim = 1, size = 4>], n = 8>
#layout_generic_b = #rotom.layout<dims = [#rotom.dim<dim = 1, size = 4>, #rotom.dim<dim = 0, size = 2>], n = 8>
#seed_generic_a = #rotom.seed<layouts = [#layout_generic_a]>
#seed_generic_b = #rotom.seed<layouts = [#layout_generic_b]>

module {
  // CHECK: func.func @generic_assign
  // CHECK-SAME: tensor<2x4xf32> {rotom.layout = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 2>, #rotom.dim<dim = 1, size = 4>], n = 8>}
  func.func @generic_assign(%arg0: tensor<2x4xf32> {rotom.seed = #seed_generic_a}, %arg1: tensor<2x4xf32> {rotom.seed = #seed_generic_b}) -> tensor<2x4xf32> {
    %empty = tensor.empty() : tensor<2x4xf32>
    // CHECK: linalg.generic
    // CHECK-SAME: rotom.layout = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 2>, #rotom.dim<dim = 1, size = 4>], n = 8>
    %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<2x4xf32>, tensor<2x4xf32>) outs(%empty : tensor<2x4xf32>) {
    ^bb0(%in0: f32, %in1: f32, %out: f32):
      %1 = arith.addf %in0, %in1 : f32
      linalg.yield %1 : f32
    } -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
}

// -----

#layout_cost_good = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 2, stride = 2>, #rotom.dim<dim = 0, size = 2>, #rotom.dim<dim = 1, size = 4>], n = 8>
#layout_cost_bad = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 4>, #rotom.dim<dim = 1, size = 2, stride = 2>, #rotom.dim<dim = 1, size = 2>], n = 8>
#seed_cost_mixed = #rotom.seed<layouts = [#layout_cost_bad, #layout_cost_good]>
#seed_cost_good = #rotom.seed<layouts = [#layout_cost_good]>

module {
  // CHECK: func.func @cost_beats_seed_order
  // CHECK-SAME: tensor<4x4xf32> {rotom.layout = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 2, stride = 2>, #rotom.dim<dim = 0, size = 2>, #rotom.dim<dim = 1, size = 4>], n = 8>}
  func.func @cost_beats_seed_order(%arg0: tensor<4x4xf32> {rotom.seed = #seed_cost_mixed}, %arg1: tensor<4x4xf32> {rotom.seed = #seed_cost_good}) -> tensor<4x4xf32> {
    // CHECK: arith.addf
    // CHECK-SAME: rotom.layout = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 2, stride = 2>, #rotom.dim<dim = 0, size = 2>, #rotom.dim<dim = 1, size = 4>], n = 8>
    // CHECK-NOT: secret.kernel
    %0 = arith.addf %arg0, %arg1 : tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}

// -----

#layout_chain_a = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 2, stride = 2>, #rotom.dim<dim = 0, size = 2>, #rotom.dim<dim = 1, size = 4>], n = 8>
#layout_chain_b = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 4>, #rotom.dim<dim = 1, size = 2, stride = 2>, #rotom.dim<dim = 1, size = 2>], n = 8>
#seed_chain_ab = #rotom.seed<layouts = [#layout_chain_a, #layout_chain_b]>
#seed_chain_b = #rotom.seed<layouts = [#layout_chain_b]>

module {
  // CHECK: func.func @downstream_selects_intermediate
  // CHECK-SAME: tensor<4x4xf32> {rotom.layout = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 4>, #rotom.dim<dim = 1, size = 2, stride = 2>, #rotom.dim<dim = 1, size = 2>], n = 8>}
  func.func @downstream_selects_intermediate(%arg0: tensor<4x4xf32> {rotom.seed = #seed_chain_ab}, %arg1: tensor<4x4xf32> {rotom.seed = #seed_chain_b}) -> tensor<4x4xf32> {
    // CHECK: tensor.extract_slice
    // CHECK-SAME: rotom.layout = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 4>, #rotom.dim<dim = 1, size = 2, stride = 2>, #rotom.dim<dim = 1, size = 2>], n = 8>
    %0 = tensor.extract_slice %arg0[0, 0] [4, 4] [1, 1] : tensor<4x4xf32> to tensor<4x4xf32>
    // CHECK: arith.addf
    // CHECK-SAME: rotom.layout = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 4>, #rotom.dim<dim = 1, size = 2, stride = 2>, #rotom.dim<dim = 1, size = 2>], n = 8>
    %1 = arith.addf %0, %arg1 : tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
}

// -----

#layout_add_align_a = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 4>, #rotom.dim<dim = 1, size = 4>], n = 16>
#layout_add_align_b = #rotom.layout<dims = [#rotom.dim<dim = 1, size = 4>, #rotom.dim<dim = 0, size = 4>], n = 16>
#seed_add_align_a = #rotom.seed<layouts = [#layout_add_align_a]>
#seed_add_align_b = #rotom.seed<layouts = [#layout_add_align_b]>

module {
  // CHECK: func.func @add_non_rolled_aligned
  // CHECK-SAME: tensor<4x4xf32> {rotom.layout = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 4>, #rotom.dim<dim = 1, size = 4>], n = 16>}
  func.func @add_non_rolled_aligned(%arg0: tensor<4x4xf32> {rotom.seed = #seed_add_align_a}, %arg1: tensor<4x4xf32> {rotom.seed = #seed_add_align_b}) -> tensor<4x4xf32> {
    // CHECK: arith.addf
    // CHECK-SAME: rotom.layout = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 4>, #rotom.dim<dim = 1, size = 4>], n = 16>
    // CHECK-SAME: secret.kernel = #secret.kernel<name = "RotomAdd"
    %0 = arith.addf %arg0, %arg1 : tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  // CHECK: func.func @mul_non_rolled_aligned
  // CHECK-SAME: tensor<4x4xf32> {rotom.layout = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 4>, #rotom.dim<dim = 1, size = 4>], n = 16>}
  func.func @mul_non_rolled_aligned(%arg0: tensor<4x4xf32> {rotom.seed = #seed_add_align_a}, %arg1: tensor<4x4xf32> {rotom.seed = #seed_add_align_b}) -> tensor<4x4xf32> {
    // CHECK: arith.mulf
    // CHECK-SAME: rotom.layout = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 4>, #rotom.dim<dim = 1, size = 4>], n = 16>
    // CHECK-SAME: secret.kernel = #secret.kernel<name = "RotomMul"
    %0 = arith.mulf %arg0, %arg1 : tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}

// -----

#layout_matmul_lhs = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 2>, #rotom.dim<dim = 1, size = 4>], n = 32>
#layout_matmul_rhs = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 4>, #rotom.dim<dim = 1, size = 8>], n = 32>
#seed_matmul_lhs = #rotom.seed<layouts = [#layout_matmul_lhs]>
#seed_matmul_rhs = #rotom.seed<layouts = [#layout_matmul_rhs]>

module {
  // CHECK: func.func @matmul_non_rolled_aligned
  // CHECK-SAME: tensor<2x8xf32> {rotom.layout = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 2>, #rotom.dim<dim = 1, size = 8>], n = 32>}
  func.func @matmul_non_rolled_aligned(%lhs: tensor<2x4xf32> {rotom.seed = #seed_matmul_lhs}, %rhs: tensor<4x8xf32> {rotom.seed = #seed_matmul_rhs}) -> tensor<2x8xf32> {
    %empty = tensor.empty() : tensor<2x8xf32>
    // CHECK: linalg.matmul
    // CHECK-SAME: rotom.layout = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 2>, #rotom.dim<dim = 1, size = 8>], n = 32>
    %0 = linalg.matmul ins(%lhs, %rhs : tensor<2x4xf32>, tensor<4x8xf32>) outs(%empty : tensor<2x8xf32>) -> tensor<2x8xf32>
    return %0 : tensor<2x8xf32>
  }
}
