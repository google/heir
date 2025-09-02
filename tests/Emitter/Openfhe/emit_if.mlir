// RUN: heir-translate %s --emit-openfhe-pke --split-input-file | FileCheck %s

!cc = !openfhe.crypto_context
!pk = !openfhe.public_key
!sk = !openfhe.private_key
module attributes {scheme.ckks} {
  // CHECK: std::vector<float> if(
  // CHECK-SAME: std::vector<float> [[v0:.*]],
  func.func @if(%cc: !cc, %0: tensor<1x1024xf32>, %sk: !sk) -> tensor<1x10xf32> {
    // CHECK-DAG: size_t [[v2:.*]] = 16;
    %c1024 = arith.constant 1024 : index
    %c16 = arith.constant 16 : index
    %c6 = arith.constant 6 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<1x10xf32>
    // CHECK: std::vector<float> [[v6:.*]](10, 0);
    // CHECK: std::vector<float> [[v7:.*]] = [[v6]];
    // CHECK: for (auto [[v8:.*]] = 0; [[v8]] < 1024; ++[[v8]])
    %1 = scf.for %arg1 = %c0 to %c1024 step %c1 iter_args(%arg2 = %cst) -> (tensor<1x10xf32>) {
      %2 = arith.addi %arg1, %c6 : index
      %3 = arith.remsi %2, %c16 : index
      %4 = arith.cmpi sge, %3, %c6 : index
      // CHECK: bool [[v12:.*]] = [[v11:.*]] >= [[v3:.*]];
      // CHECK: if ([[v12]]) {
      // CHECK:  size_t [[v14:.*]] = [[v8]] % [[v2]];
      // CHECK:  float [[v15:.*]] = [[v0]][[[v8]] + 1024 * (0)];
      // CHECK:  [[v7]][[[v14]] + 10 * (0)] = [[v15]];
      // CHECK: }
      %5 = scf.if %4 -> (tensor<1x10xf32>) {
        %6 = arith.remsi %arg1, %c16 : index
        %extracted_0 = tensor.extract %0[%c0, %arg1] : tensor<1x1024xf32>
        %inserted = tensor.insert %extracted_0 into %arg2[%c0, %6] : tensor<1x10xf32>
        scf.yield %inserted : tensor<1x10xf32>
      } else {
        scf.yield %arg2 : tensor<1x10xf32>
      }
      scf.yield %5 : tensor<1x10xf32>
    }
    return %1 : tensor<1x10xf32>
  }
}

// -----

!cc = !openfhe.crypto_context
!pk = !openfhe.public_key
!sk = !openfhe.private_key
module attributes {scheme.ckks} {
  // CHECK: int32_t if_scalar(
  // CHECK-SAME: int32_t [[v0:.*]],
  func.func @if_scalar(%cc: !cc, %arg0: i32, %sk: !sk) -> i32 {
    // CHECK-DAG: int32_t [[v1:.*]] = 6;
    // CHECK: bool [[v2:.*]] = [[v0]] >= [[v1]]
    // CHECK: int32_t [[v3:.*]];
    // CHECK: if ([[v2]]) {
    // CHECK:  int32_t [[v4:.*]] = [[v0]] % [[v1]];
    // CHECK:  [[v3]] = [[v4]];
    // CHECK: } else {
    // CHECK:  [[v3]] = [[v0]];
    // CHECK: }
    // CHECK: return [[v3]];
    %c6 = arith.constant 6 : i32
    %4 = arith.cmpi sge, %arg0, %c6 : i32
    %5 = scf.if %4 -> (i32) {
      %6 = arith.remsi %arg0, %c6 : i32
      scf.yield %6 : i32
    } else {
      scf.yield %arg0 : i32
    }
    return %5 : i32
  }
}
