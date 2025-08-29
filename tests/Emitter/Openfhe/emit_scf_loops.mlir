// RUN: heir-translate %s --emit-openfhe-pke --split-input-file | FileCheck %s

!cc = !openfhe.crypto_context
!pk = !openfhe.public_key
module attributes {scheme.ckks} {
  // CHECK: std::vector<float> for(
  // CHECK-SAME: std::vector<float> [[v0:.*]],
  func.func @for(%cc: !cc, %arg0: tensor<1x784xf32>, %pk: !pk) -> tensor<1x1024xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c784 = arith.constant 784 : index
    // CHECK: std::vector<float> [[v1:.*]](1024, 0);
    // CHECK: std::vector<float> [[v5:.*]] = [[v1]]
    // CHECK: for (auto [[v6:.*]] = 0; [[v6]] < 784; ++[[v6]])
    // CHECK:  float [[v8:.*]] = [[v0]][[[v6]] + 784 * (0)];
    // CHECK:  [[v5]][[[v6]] + 1024 * (0)] = [[v8]];
    // CHECK: }
    // CHECK: return [[v5]];
    %0 = scf.for %arg1 = %c0 to %c784 step %c1 iter_args(%arg2 = %cst) -> (tensor<1x1024xf32>) {
      %extracted = tensor.extract %arg0[%c0, %arg1] : tensor<1x784xf32>
      %inserted = tensor.insert %extracted into %arg2[%c0, %arg1] : tensor<1x1024xf32>
      scf.yield %inserted : tensor<1x1024xf32>
    }
    return %0 : tensor<1x1024xf32>
  }
}
