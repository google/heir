// RUN: heir-opt --rotate-and-reduce --canonicalize %s | FileCheck %s


// Sum all entries of a tensor into a single scalar
// CHECK-LABEL: @simple_sum
// CHECK-SAME: (%[[arg0:.*]]: tensor<8xi32>
// CHECK-NEXT: %[[c0:.*]] = arith.constant 0
// CHECK-NEXT: %[[c1:.*]] = arith.constant 1
// CHECK-NEXT: %[[c2:.*]] = arith.constant 2
// CHECK-NEXT: %[[c4:.*]] = arith.constant 4
// CHECK-NEXT: %[[v0:.*]] = tensor_ext.rotate %[[arg0]], %[[c4]]
// CHECK-NEXT: %[[v1:.*]] = arith.addi %[[arg0]], %[[v0]]
// CHECK-NEXT: %[[v2:.*]] = tensor_ext.rotate %[[v1]], %[[c2]]
// CHECK-NEXT: %[[v3:.*]] = arith.addi %[[v1]], %[[v2]]
// CHECK-NEXT: %[[v4:.*]] = tensor_ext.rotate %[[v3]], %[[c1]]
// CHECK-NEXT: %[[v5:.*]] = arith.addi %[[v3]], %[[v4]]
// CHECK-NEXT: %[[v6:.*]] = tensor.extract %[[v5]][%[[c0]]]
// CHECK-NEXT: return %[[v6]]
func.func @simple_sum(%arg0: tensor<8xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %0 = tensor.extract %arg0[%c0] : tensor<8xi32>
  %1 = tensor.extract %arg0[%c1] : tensor<8xi32>
  %2 = tensor.extract %arg0[%c2] : tensor<8xi32>
  %3 = tensor.extract %arg0[%c3] : tensor<8xi32>
  %4 = tensor.extract %arg0[%c4] : tensor<8xi32>
  %5 = tensor.extract %arg0[%c5] : tensor<8xi32>
  %6 = tensor.extract %arg0[%c6] : tensor<8xi32>
  %7 = tensor.extract %arg0[%c7] : tensor<8xi32>
  %8 = arith.addi %0, %1 : i32
  %9 = arith.addi %8, %2 : i32
  %10 = arith.addi %9, %3 : i32
  %11 = arith.addi %10, %4 : i32
  %12 = arith.addi %11, %5 : i32
  %13 = arith.addi %12, %6 : i32
  %14 = arith.addi %13, %7 : i32
  return %14 : i32
}

// CHECK-LABEL: @not_supported_mixed_ops
// CHECK-NOT: tensor_ext.rotate
func.func @not_supported_mixed_ops(%arg0: tensor<8xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %0 = tensor.extract %arg0[%c0] : tensor<8xi32>
  %1 = tensor.extract %arg0[%c1] : tensor<8xi32>
  %2 = tensor.extract %arg0[%c2] : tensor<8xi32>
  %3 = tensor.extract %arg0[%c3] : tensor<8xi32>
  %4 = tensor.extract %arg0[%c4] : tensor<8xi32>
  %5 = tensor.extract %arg0[%c5] : tensor<8xi32>
  %6 = tensor.extract %arg0[%c6] : tensor<8xi32>
  %7 = tensor.extract %arg0[%c7] : tensor<8xi32>
  %8 = arith.addi %0, %1 : i32
  %9 = arith.addi %8, %2 : i32
  %10 = arith.addi %9, %3 : i32
  %11 = arith.muli %10, %4 : i32
  %12 = arith.addi %11, %5 : i32
  %13 = arith.addi %12, %6 : i32
  %14 = arith.addi %13, %7 : i32
  return %14 : i32
}

// CHECK-LABEL: @not_supported_missing_indices
// CHECK-NOT: tensor_ext.rotate
func.func @not_supported_missing_indices(%arg0: tensor<16xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %c8 = arith.constant 8 : index
  %c9 = arith.constant 9 : index
  %c10 = arith.constant 10 : index
  %c11 = arith.constant 11 : index
  %c12 = arith.constant 12 : index
  %c13 = arith.constant 13 : index
  %c14 = arith.constant 14 : index
  %0 = tensor.extract %arg0[%c0] : tensor<16xi32>
  %1 = tensor.extract %arg0[%c1] : tensor<16xi32>
  %2 = tensor.extract %arg0[%c2] : tensor<16xi32>
  %3 = tensor.extract %arg0[%c3] : tensor<16xi32>
  %4 = tensor.extract %arg0[%c4] : tensor<16xi32>
  %5 = tensor.extract %arg0[%c5] : tensor<16xi32>
  %6 = tensor.extract %arg0[%c6] : tensor<16xi32>
  %7 = tensor.extract %arg0[%c7] : tensor<16xi32>
  %8 = tensor.extract %arg0[%c8] : tensor<16xi32>
  %9 = tensor.extract %arg0[%c9] : tensor<16xi32>
  %10 = tensor.extract %arg0[%c10] : tensor<16xi32>
  %11 = tensor.extract %arg0[%c11] : tensor<16xi32>
  %12 = tensor.extract %arg0[%c12] : tensor<16xi32>
  %13 = tensor.extract %arg0[%c13] : tensor<16xi32>
  %14 = tensor.extract %arg0[%c14] : tensor<16xi32>
  // missing element 15
  %v1 = arith.addi %0, %1 : i32
  %v2 = arith.addi %v1, %2 : i32
  %v3 = arith.addi %v2, %3 : i32
  %v4 = arith.addi %v3, %4 : i32
  %v5 = arith.addi %v4, %5 : i32
  %v6 = arith.addi %v5, %6 : i32
  %v7 = arith.addi %v6, %7 : i32
  %v8 = arith.addi %v7, %8 : i32
  %v9 = arith.addi %v8, %9 : i32
  %v10 = arith.addi %v9, %10 : i32
  %v11 = arith.addi %v10, %11 : i32
  %v12 = arith.addi %v11, %12 : i32
  %v13 = arith.addi %v12, %13 : i32
  %v14 = arith.addi %v13, %14 : i32
  return %v14 : i32
}

// CHECK-LABEL: @not_supported_repeated_indices
// CHECK-NOT: tensor_ext.rotate
func.func @not_supported_repeated_indices(%arg0: tensor<8xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %0 = tensor.extract %arg0[%c0] : tensor<8xi32>
  %1 = tensor.extract %arg0[%c1] : tensor<8xi32>
  %2 = tensor.extract %arg0[%c2] : tensor<8xi32>
  %3 = tensor.extract %arg0[%c3] : tensor<8xi32>
  // repeats element 3
  %4 = tensor.extract %arg0[%c3] : tensor<8xi32>
  %5 = tensor.extract %arg0[%c5] : tensor<8xi32>
  %6 = tensor.extract %arg0[%c6] : tensor<8xi32>
  %7 = tensor.extract %arg0[%c7] : tensor<8xi32>
  %8 = arith.addi %0, %1 : i32
  %9 = arith.addi %8, %2 : i32
  %10 = arith.addi %9, %3 : i32
  %11 = arith.addi %10, %4 : i32
  %12 = arith.addi %11, %5 : i32
  %13 = arith.addi %12, %6 : i32
  %14 = arith.addi %13, %7 : i32
  return %14 : i32
}

// CHECK-LABEL: @not_supported_unsupported_op
// CHECK-NOT: tensor_ext.rotate
func.func @not_supported_unsupported_op(%arg0: tensor<8xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %0 = tensor.extract %arg0[%c0] : tensor<8xi32>
  %1 = tensor.extract %arg0[%c1] : tensor<8xi32>
  %2 = tensor.extract %arg0[%c2] : tensor<8xi32>
  %3 = tensor.extract %arg0[%c3] : tensor<8xi32>
  %4 = tensor.extract %arg0[%c4] : tensor<8xi32>
  %5 = tensor.extract %arg0[%c5] : tensor<8xi32>
  %6 = tensor.extract %arg0[%c6] : tensor<8xi32>
  %7 = tensor.extract %arg0[%c7] : tensor<8xi32>
  %8 = arith.subi %0, %1 : i32
  %9 = arith.subi %8, %2 : i32
  %10 = arith.subi %9, %3 : i32
  %11 = arith.subi %10, %4 : i32
  %12 = arith.subi %11, %5 : i32
  %13 = arith.subi %12, %6 : i32
  %14 = arith.subi %13, %7 : i32
  return %14 : i32
}

// 2D tensor not supported
// CHECK-LABEL: @not_supported_bad_tensor_shape
// CHECK-NOT: tensor_ext.rotate
func.func @not_supported_bad_tensor_shape(%arg0: tensor<1x8xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %0 = tensor.extract %arg0[%c1, %c0] : tensor<1x8xi32>
  %1 = tensor.extract %arg0[%c1, %c1] : tensor<1x8xi32>
  %2 = tensor.extract %arg0[%c1, %c2] : tensor<1x8xi32>
  %3 = tensor.extract %arg0[%c1, %c3] : tensor<1x8xi32>
  %4 = tensor.extract %arg0[%c1, %c4] : tensor<1x8xi32>
  %5 = tensor.extract %arg0[%c1, %c5] : tensor<1x8xi32>
  %6 = tensor.extract %arg0[%c1, %c6] : tensor<1x8xi32>
  %7 = tensor.extract %arg0[%c1, %c7] : tensor<1x8xi32>
  %8 = arith.addi %0, %1 : i32
  %9 = arith.addi %8, %2 : i32
  %10 = arith.addi %9, %3 : i32
  %11 = arith.addi %10, %4 : i32
  %12 = arith.addi %11, %5 : i32
  %13 = arith.addi %12, %6 : i32
  %14 = arith.addi %13, %7 : i32
  return %14 : i32
}

// reducing from multiple input tensors
// CHECK-LABEL: @not_supported_multiple_tensors
// CHECK-NOT: tensor_ext.rotate
func.func @not_supported_multiple_tensors(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %0 = tensor.extract %arg0[%c0] : tensor<8xi32>
  %1 = tensor.extract %arg0[%c1] : tensor<8xi32>
  // uses %arg1
  %2 = tensor.extract %arg1[%c2] : tensor<8xi32>
  %3 = tensor.extract %arg0[%c3] : tensor<8xi32>
  %4 = tensor.extract %arg0[%c4] : tensor<8xi32>
  %5 = tensor.extract %arg0[%c5] : tensor<8xi32>
  %6 = tensor.extract %arg0[%c6] : tensor<8xi32>
  %7 = tensor.extract %arg0[%c7] : tensor<8xi32>
  %8 = arith.addi %0, %1 : i32
  %9 = arith.addi %8, %2 : i32
  %10 = arith.addi %9, %3 : i32
  %11 = arith.addi %10, %4 : i32
  %12 = arith.addi %11, %5 : i32
  %13 = arith.addi %12, %6 : i32
  %14 = arith.addi %13, %7 : i32
  return %14 : i32
}

// CHECK-LABEL: @not_supported_non_constant_index_access
// CHECK-NOT: tensor_ext.rotate
func.func @not_supported_non_constant_index_access(%arg0: tensor<8xi32>, %arg1: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %0 = tensor.extract %arg0[%c0] : tensor<8xi32>
  %1 = tensor.extract %arg0[%c1] : tensor<8xi32>
  // uses non-constant index
  %2 = tensor.extract %arg0[%arg1] : tensor<8xi32>
  %3 = tensor.extract %arg0[%c3] : tensor<8xi32>
  %4 = tensor.extract %arg0[%c4] : tensor<8xi32>
  %5 = tensor.extract %arg0[%c5] : tensor<8xi32>
  %6 = tensor.extract %arg0[%c6] : tensor<8xi32>
  %7 = tensor.extract %arg0[%c7] : tensor<8xi32>
  %8 = arith.addi %0, %1 : i32
  %9 = arith.addi %8, %2 : i32
  %10 = arith.addi %9, %3 : i32
  %11 = arith.addi %10, %4 : i32
  %12 = arith.addi %11, %5 : i32
  %13 = arith.addi %12, %6 : i32
  %14 = arith.addi %13, %7 : i32
  return %14 : i32
}

// CHECK-LABEL: @not_supported_non_tensor_operands
// CHECK-NOT: tensor_ext.rotate
// TODO(#522): support this
func.func @not_supported_non_tensor_operands(%arg0: tensor<8xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %c2_i32 = arith.constant 2 : i32
  %0 = tensor.extract %arg0[%c0] : tensor<8xi32>
  %1 = tensor.extract %arg0[%c1] : tensor<8xi32>
  %2 = tensor.extract %arg0[%c2] : tensor<8xi32>
  %3 = tensor.extract %arg0[%c3] : tensor<8xi32>
  %4 = tensor.extract %arg0[%c4] : tensor<8xi32>
  %5 = tensor.extract %arg0[%c5] : tensor<8xi32>
  %6 = tensor.extract %arg0[%c6] : tensor<8xi32>
  %7 = tensor.extract %arg0[%c7] : tensor<8xi32>
  %8 = arith.addi %0, %1 : i32
  // next op uses non-tensor operand
  %9 = arith.addi %8, %c2_i32 : i32
  %10 = arith.addi %9, %3 : i32
  %11 = arith.addi %10, %4 : i32
  %12 = arith.addi %11, %5 : i32
  %13 = arith.addi %12, %6 : i32
  %14 = arith.addi %13, %7 : i32
  %15 = arith.addi %14, %2 : i32
  return %15 : i32
}
