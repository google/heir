// RUN: heir-opt --lower-unpack --canonicalize %s | FileCheck %s

// Tests `tensor_ext.unpack` with a dense-permutation layout attribute

// -----------------------------------------------------------------------------
// Case 1: single dense permutation with a rank-1 data-semantic target.
//
// The permutation `[[0, 3, 0, 0], [0, 5, 0, 1], [0, 1, 0, 2]]` gathers slots
// 3, 5, 1 from a single ciphertext view (rank-2, one row) into positions
// 0, 1, 2 of a rank-1 result.

#dense_perm_single = dense<[[0, 3, 0, 0], [0, 5, 0, 1], [0, 1, 0, 2]]>
    : tensor<3x4xi64>
#orig_single = #tensor_ext.original_type<
    originalType = tensor<3xi32>, layout = #dense_perm_single>

// CHECK: @unpack_dense_single
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:   %[[ZERO:.*]] = arith.constant dense<0> : tensor<3xi32>
// CHECK-DAG:   %[[E0:.*]] = tensor.extract %arg0[%[[C0]], %[[C3]]] : tensor<1x8xi32>
// CHECK-DAG:   tensor.insert %[[E0]] into %{{.*}}[%[[C0]]] : tensor<3xi32>
// CHECK-DAG:   %[[E1:.*]] = tensor.extract %arg0[%[[C0]], %[[C5]]] : tensor<1x8xi32>
// CHECK-DAG:   tensor.insert %[[E1]] into %{{.*}}[%[[C1]]] : tensor<3xi32>
// CHECK-DAG:   %[[E2:.*]] = tensor.extract %arg0[%[[C0]], %[[C1]]] : tensor<1x8xi32>
// CHECK-DAG:   tensor.insert %[[E2]] into %{{.*}}[%[[C2]]] : tensor<3xi32>
// CHECK:       return %{{.*}} : tensor<3xi32>
func.func @unpack_dense_single(
    %arg0: tensor<1x8xi32> {tensor_ext.original_type = #orig_single}
) -> tensor<3xi32> {
  %0 = tensor_ext.unpack %arg0 {layout = #dense_perm_single}
      : (tensor<1x8xi32>) -> tensor<3xi32>
  return %0 : tensor<3xi32>
}

// -----------------------------------------------------------------------------
// Case 2: dense permutation with cross-ct source. The permutation
// `[[1, 2, 0, 0], [0, 4, 0, 1]]` says logical output 0 lives at (ct=1, slot=2)
// and logical output 1 lives at (ct=0, slot=4).

#dense_perm_cross = dense<[[1, 2, 0, 0], [0, 4, 0, 1]]> : tensor<2x4xi64>
#orig_cross = #tensor_ext.original_type<
    originalType = tensor<2xi32>, layout = #dense_perm_cross>

// CHECK: @unpack_dense_cross_ct
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[E0:.*]] = tensor.extract %arg0[%[[C1]], %[[C2]]] : tensor<2x8xi32>
// CHECK-DAG:   tensor.insert %[[E0]] into %{{.*}}[%[[C0]]] : tensor<2xi32>
// CHECK-DAG:   %[[E1:.*]] = tensor.extract %arg0[%[[C0]], %[[C4]]] : tensor<2x8xi32>
// CHECK-DAG:   tensor.insert %[[E1]] into %{{.*}}[%[[C1]]] : tensor<2xi32>
// CHECK:       return %{{.*}} : tensor<2xi32>
func.func @unpack_dense_cross_ct(
    %arg0: tensor<2x8xi32> {tensor_ext.original_type = #orig_cross}
) -> tensor<2xi32> {
  %0 = tensor_ext.unpack %arg0 {layout = #dense_perm_cross}
      : (tensor<2x8xi32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}
