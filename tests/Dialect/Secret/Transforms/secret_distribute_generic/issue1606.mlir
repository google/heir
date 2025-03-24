// A regression test for https://github.com/google/heir/issues/1606

// RUN: heir-opt --secret-distribute-generic %s

module {
  func.func @hv_matmul(%arg0: !secret.secret<tensor<1024xf32>>) -> !secret.secret<tensor<1024xf32>> {
    %c1 = arith.constant 1 : index
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1024xf32>>) {
    ^body(%input0: tensor<1024xf32>):
      %1:2 = affine.for %arg1 = 1 to 1024 iter_args(%arg2 = %input0, %arg3 = %input0) -> (tensor<1024xf32>, tensor<1024xf32>) {
        %2 = tensor_ext.rotate %arg3, %c1 : tensor<1024xf32>, index
        %3 = arith.addf %arg2, %2 : tensor<1024xf32>
        affine.yield %3, %2 : tensor<1024xf32>, tensor<1024xf32>
      }
      secret.yield %1#0 : tensor<1024xf32>
    } -> !secret.secret<tensor<1024xf32>>
    return %0 : !secret.secret<tensor<1024xf32>>
  }
}
