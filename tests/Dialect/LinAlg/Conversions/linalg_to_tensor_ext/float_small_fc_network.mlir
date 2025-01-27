// This test verifies that a small fully connected network lowers without returning
// an error.
// TODO: write a test that verifies the correctness of the lowering.

// RUN: heir-opt %s --linalg-to-tensor-ext=tiling-size=4 --tosa-to-secret-arith --canonicalize | FileCheck %s

// CHECK:        func.func @test_float_small_fc_network(%[[ARG:.*]]: !secret.secret<tensor<1x4xf32>>)
module {
func.func @test_float_small_fc_network(%input : !secret.secret<tensor<1x1xf32>>) -> !secret.secret<tensor<1x1xf32>> {
  %matrix1 = arith.constant dense<[[1.0, 2.0, 3.0, 4.0]]> : tensor<1x4xf32>
  %bias1 = arith.constant dense<[[5.0, 6.0, 7.0, 8.0]]> : tensor<1x4xf32>
  %layer1 = secret.generic ins (%input : !secret.secret<tensor<1x1xf32>>) {
  ^bb0(%converted_input1: tensor<1x1xf32>):
    %0 = linalg.matmul ins(%converted_input1, %matrix1 : tensor<1x1xf32>, tensor<1x4xf32>) outs(%bias1 : tensor<1x4xf32>) -> tensor<1x4xf32>
    secret.yield %0 : tensor<1x4xf32>
  } -> !secret.secret<tensor<1x4xf32>>

  %activation_layer1 = secret.generic ins (%layer1 : !secret.secret<tensor<1x4xf32>>) {
  ^bb0(%converted_activation_layer_vec1: tensor<1x4xf32>):
    %0 = tosa.sigmoid %converted_activation_layer_vec1 : (tensor<1x4xf32>) -> tensor<1x4xf32>
    secret.yield %0 : tensor<1x4xf32>
  } -> !secret.secret<tensor<1x4xf32>>

  %matrix2 = arith.constant dense<[[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0], [90.0, 100.0, 110.0, 120.0], [130.0, 140.0, 150.0, 160.0]]> : tensor<4x4xf32>
  %bias2 = arith.constant dense<[[170.0, 180.0, 190.0, 200.0]]> : tensor<1x4xf32>
  %layer2 = secret.generic ins (%layer1 : !secret.secret<tensor<1x4xf32>>) {
  ^bb0(%converted_vec2: tensor<1x4xf32>):
    %1 = linalg.matmul ins(%converted_vec2, %matrix2 : tensor<1x4xf32>, tensor<4x4xf32>) outs(%bias2 : tensor<1x4xf32>) -> tensor<1x4xf32>
    secret.yield %1 : tensor<1x4xf32>
  } -> !secret.secret<tensor<1x4xf32>>

  %activation_layer2 = secret.generic ins (%layer2 : !secret.secret<tensor<1x4xf32>>) {
  ^bb0(%converted_activation_layer_vec2: tensor<1x4xf32>):
    %0 = tosa.sigmoid %converted_activation_layer_vec2 : (tensor<1x4xf32>) -> tensor<1x4xf32>
    secret.yield %0 : tensor<1x4xf32>
  } -> !secret.secret<tensor<1x4xf32>>

  %matrix3 = arith.constant dense<[[100.0], [200.0], [300.0], [400.0]]> : tensor<4x1xf32>
  %bias3 = arith.constant dense<[[500.0]]> : tensor<1x1xf32>
  %layer3 = secret.generic ins (%activation_layer2 : !secret.secret<tensor<1x4xf32>>) {
  ^bb0(%converted_vec3: tensor<1x4xf32>):
    %0 = linalg.matmul ins(%converted_vec3, %matrix3 : tensor<1x4xf32>, tensor<4x1xf32>) outs(%bias3 : tensor<1x1xf32>) -> tensor<1x1xf32>
    secret.yield %0 : tensor<1x1xf32>
  } -> !secret.secret<tensor<1x1xf32>>
  return %layer3 : !secret.secret<tensor<1x1xf32>>
}
}
