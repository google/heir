// RUN: heir-opt --propagate-padding %s | FileCheck %s

// CHECK-DAG: #[[$P44_57:padding[0-9]*]] = #tensor_ext.padding<logicalShape = [4, 4], paddedShape = [5, 7], zeroPadded = true>
// CHECK-DAG: #[[$P44_75:padding[0-9]*]] = #tensor_ext.padding<logicalShape = [4, 4], paddedShape = [7, 5], zeroPadded = true>
// CHECK-DAG: #[[$P44_55:padding[0-9]*]] = #tensor_ext.padding<logicalShape = [4, 4], paddedShape = [5, 5], zeroPadded = true>
// CHECK-DAG: #[[$P44_57_NZ:padding[0-9]*]] = #tensor_ext.padding<logicalShape = [4, 4], paddedShape = [5, 7], zeroPadded = false>
// CHECK-DAG: #[[$P3D:padding[0-9]*]] = #tensor_ext.padding<logicalShape = [2, 4, 4], paddedShape = [2, 5, 5], zeroPadded = true>
// CHECK-DAG: #[[$P2D_RED:padding[0-9]*]] = #tensor_ext.padding<logicalShape = [2, 4], paddedShape = [2, 5], zeroPadded = true>

// -----------------------------------------------------------------------
// Seed at tensor.pad; transfer through matmul.

// CHECK: @seed_and_matmul
func.func @seed_and_matmul(%lhs: tensor<4x4xf32>, %rhs: tensor<4x4xf32>) -> tensor<5x5xf32> {
  %cst = arith.constant 0.0 : f32
  // CHECK: tensor.pad
  // CHECK: tensor_ext.padding = #[[$P44_57]]
  %lhs_pad = tensor.pad %lhs low[0, 0] high[1, 3] {
  ^bb0(%i: index, %j: index):
    tensor.yield %cst : f32
  } : tensor<4x4xf32> to tensor<5x7xf32>
  // CHECK: tensor.pad
  // CHECK: tensor_ext.padding = #[[$P44_75]]
  %rhs_pad = tensor.pad %rhs low[0, 0] high[3, 1] {
  ^bb0(%i: index, %j: index):
    tensor.yield %cst : f32
  } : tensor<4x4xf32> to tensor<7x5xf32>
  %init = arith.constant dense<0.0> : tensor<5x5xf32>
  // CHECK: linalg.matmul
  // CHECK-SAME: tensor_ext.padding = #[[$P44_55]]
  %0 = linalg.matmul ins(%lhs_pad, %rhs_pad : tensor<5x7xf32>, tensor<7x5xf32>)
      outs(%init : tensor<5x5xf32>) -> tensor<5x5xf32>
  return %0 : tensor<5x5xf32>
}

// -----------------------------------------------------------------------
// Elementwise: scalar-splat multiply keeps zero pads; exp destroys them but
// keeps the shape claim.

// CHECK: @elementwise
func.func @elementwise(%x: tensor<4x4xf32>) -> tensor<5x7xf32> {
  %cst = arith.constant 0.0 : f32
  %x_pad = tensor.pad %x low[0, 0] high[1, 3] {
  ^bb0(%i: index, %j: index):
    tensor.yield %cst : f32
  } : tensor<4x4xf32> to tensor<5x7xf32>
  %scale = arith.constant dense<2.5e-1> : tensor<5x7xf32>
  // CHECK: arith.mulf
  // CHECK-SAME: tensor_ext.padding = #[[$P44_57]]
  %scaled = arith.mulf %x_pad, %scale : tensor<5x7xf32>
  // CHECK: math.exp
  // CHECK-SAME: tensor_ext.padding = #[[$P44_57_NZ]]
  %e = math.exp %scaled : tensor<5x7xf32>
  return %e : tensor<5x7xf32>
}

// -----------------------------------------------------------------------
// Batch matmul composes batch logical dims; reduce over the padded (zero)
// last dimension drops it.

// CHECK: @reduce_and_batch
func.func @reduce_and_batch(%a: tensor<2x4x4xf32>, %b: tensor<2x4x4xf32>) -> tensor<2x5xf32> {
  %cst = arith.constant 0.0 : f32
  %a_pad = tensor.pad %a low[0, 0, 0] high[0, 1, 1] {
  ^bb0(%i: index, %j: index, %k: index):
    tensor.yield %cst : f32
  } : tensor<2x4x4xf32> to tensor<2x5x5xf32>
  %b_pad = tensor.pad %b low[0, 0, 0] high[0, 1, 1] {
  ^bb0(%i: index, %j: index, %k: index):
    tensor.yield %cst : f32
  } : tensor<2x4x4xf32> to tensor<2x5x5xf32>
  %init = arith.constant dense<0.0> : tensor<2x5x5xf32>
  // CHECK: linalg.batch_matmul
  // CHECK-SAME: tensor_ext.padding = #[[$P3D]]
  %mm = linalg.batch_matmul ins(%a_pad, %b_pad : tensor<2x5x5xf32>, tensor<2x5x5xf32>)
      outs(%init : tensor<2x5x5xf32>) -> tensor<2x5x5xf32>
  %rinit = arith.constant dense<0.0> : tensor<2x5xf32>
  // CHECK: linalg.reduce
  // CHECK: tensor_ext.padding = #[[$P2D_RED]]
  %r = linalg.reduce ins(%mm : tensor<2x5x5xf32>) outs(%rinit : tensor<2x5xf32>) dimensions = [2]
    (%in: f32, %out: f32) {
      %s = arith.addf %in, %out : f32
      linalg.yield %s : f32
    }
  return %r : tensor<2x5xf32>
}

// -----------------------------------------------------------------------
// Negative cases: nonzero pad value gives zeroPadded = false and matmul
// refuses to propagate; low-padding stops propagation entirely.

// CHECK: @nonzero_pad_value
func.func @nonzero_pad_value(%x: tensor<4x4xf32>, %y: tensor<7x5xf32>) -> tensor<5x5xf32> {
  %cst = arith.constant 1.0 : f32
  // CHECK: tensor.pad
  // CHECK: tensor_ext.padding = #[[$P44_57_NZ]]
  %x_pad = tensor.pad %x low[0, 0] high[1, 3] {
  ^bb0(%i: index, %j: index):
    tensor.yield %cst : f32
  } : tensor<4x4xf32> to tensor<5x7xf32>
  %init = arith.constant dense<0.0> : tensor<5x5xf32>
  // CHECK: linalg.matmul
  // CHECK-NOT: tensor_ext.padding
  %0 = linalg.matmul ins(%x_pad, %y : tensor<5x7xf32>, tensor<7x5xf32>)
      outs(%init : tensor<5x5xf32>) -> tensor<5x5xf32>
  return %0 : tensor<5x5xf32>
}

// CHECK: @low_padding_stops
func.func @low_padding_stops(%x: tensor<4x4xf32>) -> tensor<5x7xf32> {
  %cst = arith.constant 0.0 : f32
  // CHECK: tensor.pad
  // CHECK-NOT: tensor_ext.padding
  %x_pad = tensor.pad %x low[1, 0] high[0, 3] {
  ^bb0(%i: index, %j: index):
    tensor.yield %cst : f32
  } : tensor<4x4xf32> to tensor<5x7xf32>
  return %x_pad : tensor<5x7xf32>
}
