// RUN: heir-opt --mlir-to-ckks="min-slot-count=8 enable-split-preprocessing" %s | FileCheck %s

// Pipeline test verifying that mlir-to-ckks with split preprocessing decomposes
// a linalg.matvec into cleartext preparation hoisted into preprocessing storage,
// and ciphertext evaluation in the preprocessed function.
// See prepare_linear_transform.mlir for the unit test of --split-preprocessing.

// CHECK: func.func @matvec__preprocessing() -> !preprocessing.storage<
// CHECK: %[[PREPARE:.*]] = kernel.prepare_linear_transform %{{.*}} <diagonal_indices = [0, 1, 2, 3]>
// CHECK: preprocessing.store %[[PREPARE]], %{{.*}} site 0<

// CHECK: func.func @matvec__preprocessed(
// CHECK: %[[LOAD:.*]] = preprocessing.load %{{.*}}[] site 0<
// CHECK: %[[EXTRACT:.*]] = tensor.extract %{{.*}}[%{{.*}}]
// CHECK: kernel.apply_linear_transform %[[EXTRACT]], %[[LOAD]] <diagonal_indices = [0, 1, 2, 3]

module attributes {backend.lattigo, scheme.ckks} {
  func.func @matvec(%arg0 : tensor<4xf32> {secret.secret}) -> tensor<4xf32> {
    %matrix = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]> : tensor<4x4xf32>
    %out = tensor.empty() : tensor<4xf32>
    %0 = linalg.matvec ins(%matrix, %arg0 : tensor<4x4xf32>, tensor<4xf32>) outs(%out : tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
