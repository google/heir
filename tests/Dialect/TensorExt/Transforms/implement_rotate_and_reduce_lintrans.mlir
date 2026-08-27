// RUN: heir-opt --split-input-file --implement-rotate-and-reduce %s | FileCheck %s

// A rotate_and_reduce marked tensor_ext.lintrans becomes a
// kernel.linear_transform when the target evaluates linear transforms
// directly. The diagonal_indices name a subset of the packed matrix's rows,
// so the named rows are gathered into a compact diagonals operand whose row k
// is the diagonal named by indices[k]. For a constant the gather folds
// immediately.

// CHECK: @lintrans_constant_subset
// CHECK{LITERAL}: dense<[[1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], [3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00]]> : tensor<2x4xf32>
// CHECK: kernel.linear_transform %{{.*}}, %{{.*}} {diagonal_indices = array<i64: 0, 2>} : tensor<4xf32>, tensor<2x4xf32> -> tensor<4xf32>
module attributes {
  backend.openfhe,
  backend.config_override = {has_kernel_linear_transform = true}
} {
  func.func @lintrans_constant_subset(%v: tensor<4xf32>) -> tensor<4xf32> {
    %matrix = arith.constant dense<[[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0], [4.0, 4.0, 4.0, 4.0]]> : tensor<4x4xf32>
    %0 = tensor_ext.rotate_and_reduce %v, %matrix {
      period = 1 : index, steps = 4 : index, reduceOp = "arith.addf",
      tensor_ext.lintrans, tensor_ext.diagonal_indices = array<i32: 0, 2>
    } : (tensor<4xf32>, tensor<4x4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// -----

// Non-constant diagonals retain their source tensor and carry the selected
// rows into linear-transform preparation.

// CHECK: @lintrans_value_subset
// CHECK-NOT: tensor.extract_slice
// CHECK-NOT: tensor.concat
// CHECK: kernel.linear_transform %{{.*}}, %{{.*}} {diagonal_indices = array<i64: 1, 3>, source_row_indices = array<i64: 1, 3>} : tensor<4xf32>, tensor<4x4xf32> -> tensor<4xf32>
module attributes {
  backend.openfhe,
  backend.config_override = {has_kernel_linear_transform = true}
} {
  func.func @lintrans_value_subset(%v: tensor<4xf32>, %matrix: tensor<4x4xf32>) -> tensor<4xf32> {
    %0 = tensor_ext.rotate_and_reduce %v, %matrix {
      period = 1 : index, steps = 4 : index, reduceOp = "arith.addf",
      tensor_ext.lintrans, tensor_ext.diagonal_indices = array<i32: 1, 3>
    } : (tensor<4xf32>, tensor<4x4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// -----

// With no diagonal_indices the rows are already positional: all diagonals are
// present and the operand passes through untouched.

// CHECK: @lintrans_all_rows
// CHECK-NOT: tensor.extract_slice
// CHECK: kernel.linear_transform %{{.*}}, %{{.*}} {diagonal_indices = array<i64: 0, 1, 2, 3>} : tensor<4xf32>, tensor<4x4xf32> -> tensor<4xf32>
module attributes {
  backend.openfhe,
  backend.config_override = {has_kernel_linear_transform = true}
} {
  func.func @lintrans_all_rows(%v: tensor<4xf32>, %matrix: tensor<4x4xf32>) -> tensor<4xf32> {
    %0 = tensor_ext.rotate_and_reduce %v, %matrix {
      period = 1 : index, steps = 4 : index, reduceOp = "arith.addf",
      tensor_ext.lintrans
    } : (tensor<4xf32>, tensor<4x4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// -----

// A resource-backed constant folds too, into a compact resource rather than a
// dense attribute.

// CHECK: @lintrans_resource_subset
// CHECK-NOT: tensor.extract_slice
// CHECK: %[[DIAGS:.*]] = arith.constant dense_resource<matrix_weights_gathered> : tensor<2x4xf32>
// CHECK: kernel.linear_transform %{{.*}}, %[[DIAGS]] {diagonal_indices = array<i64: 1, 3>} : tensor<4xf32>, tensor<2x4xf32> -> tensor<4xf32>
// Rows 1 and 3 of the blob, which hold 2.0 and 4.0, and nothing else. A
// mis-sliced row shows up here as a changed byte rather than as a missing op.
// CHECK: matrix_weights_gathered: "0x040000000000004000000040000000400000004000008040000080400000804000008040"
module attributes {
  backend.openfhe,
  backend.config_override = {has_kernel_linear_transform = true}
} {
  func.func @lintrans_resource_subset(%v: tensor<4xf32>) -> tensor<4xf32> {
    %matrix = arith.constant dense_resource<matrix_weights> : tensor<4x4xf32>
    %0 = tensor_ext.rotate_and_reduce %v, %matrix {
      period = 1 : index, steps = 4 : index, reduceOp = "arith.addf",
      tensor_ext.lintrans, tensor_ext.diagonal_indices = array<i32: 1, 3>
    } : (tensor<4xf32>, tensor<4x4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      matrix_weights: "0x040000000000803f0000803f0000803f0000803f000000400000004000000040000000400000404000004040000040400000404000008040000080400000804000008040"
    }
  }
#-}
