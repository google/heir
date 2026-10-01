// RUN: heir-opt --canonicalize %s | FileCheck %s

#map = affine_map<(d0) -> (d0)>
module {
  // CHECK: func.func @generic_using_expand(%[[ARG0:.+]]: !secret.secret<tensor<1x13xf32>>) -> !secret.secret<tensor<1x1xf32>> {
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[CST_DUMMY2:.+]] = arith.constant dense<0.000000e+00> : tensor<32x29xf32>
  // CHECK-DAG: %[[CST_DUMMY1:.+]] = arith.constant dense<0.000000e+00> : tensor<27x32xf32>
  // CHECK-DAG: %[[CST_0:.+]] = arith.constant dense<0.000000e+00> : tensor<27x29xf32>
  // CHECK-DAG: %[[CST_22:.+]] = arith.constant dense<0> : tensor<351xi64>
  // CHECK: %[[MATMUL:.+]] = linalg.matmul ins(%[[CST_DUMMY1]], %[[CST_DUMMY2]] : tensor<27x32xf32>, tensor<32x29xf32>) outs(%[[CST_0]] : tensor<27x29xf32>) -> tensor<27x29xf32>
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<351xf32>
  // CHECK: %[[EXPANDED_87:.+]] = tensor.expand_shape %[[MATMUL]] {{\[\[}}0, 1], [2]] output_shape [1, 27, 29] : tensor<27x29xf32> into tensor<1x27x29xf32>
  // CHECK: %[[LINALG_GEN:.+]] = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%[[CST_22]], %[[CST_22]], %[[CST_22]] : tensor<351xi64>, tensor<351xi64>, tensor<351xi64>) outs(%[[EMPTY]] : tensor<351xf32>) {
  // CHECK: ^bb0(%{{.+}}: i64, %{{.+}}: i64, %{{.+}}: i64, %{{.+}}: f32):
  // CHECK:   %[[EXTRACTED:.+]] = tensor.extract %[[EXPANDED_87]][%[[C0]], %[[C0]], %[[C0]]] : tensor<1x27x29xf32>
  // CHECK:   linalg.yield %[[EXTRACTED]] : f32
  // CHECK: }
  // CHECK: %[[EXPANDED_95:.+]] = tensor.expand_shape %[[LINALG_GEN]] {{\[\[}}0, 1]] output_shape [1, 351] : tensor<351xf32> into tensor<1x351xf32>
  // CHECK: %[[RES_SLICE:.+]] = tensor.extract_slice %[[EXPANDED_95]][0, 0] [1, 1] [1, 1] : tensor<1x351xf32> to tensor<1x1xf32>
  // CHECK: %[[SEC_GEN:.+]] = secret.generic(%[[ARG0]]{{.*}}) {
  // CHECK: ^body(%[[INPUT0:.+]]: tensor<1x13xf32>):
  // CHECK:   %[[SLICE0:.+]] = tensor.extract_slice %[[INPUT0]][0, 0] [1, 1] [1, 1] : tensor<1x13xf32> to tensor<1x1xf32>
  // CHECK:   %[[FINAL_RES:.+]] = arith.addf %[[SLICE0]], %[[RES_SLICE]] : tensor<1x1xf32>
  // CHECK:   secret.yield %[[FINAL_RES]] : tensor<1x1xf32>
  // CHECK: }
  // CHECK: return %[[SEC_GEN]]
  func.func @generic_using_expand(%arg0: !secret.secret<tensor<1x13xf32>>) -> !secret.secret<tensor<1x1xf32>> {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<27x29xf32>
    %cst_22 = arith.constant dense<0> : tensor<351xi64>
    %0 = secret.generic(%arg0: !secret.secret<tensor<1x13xf32>>) {
    ^body(%input0: tensor<1x13xf32>):
      %cst_dummy1 = arith.constant dense<0.000000e+00> : tensor<27x32xf32>
      %cst_dummy2 = arith.constant dense<0.000000e+00> : tensor<32x29xf32>
      %10 = linalg.matmul ins(%cst_dummy1, %cst_dummy2 : tensor<27x32xf32>, tensor<32x29xf32>) outs(%cst_0 : tensor<27x29xf32>) -> tensor<27x29xf32>
      %expanded_87 = tensor.expand_shape %10 [[0, 1], [2]] output_shape [1, 27, 29] : tensor<27x29xf32> into tensor<1x27x29xf32>
      %25 = tensor.empty() : tensor<351xf32>

      // This generic uses expanded_87 which is defined outside the
      // linalg.generic, so the hoisting needs to properly account for
      // dominance.
      %26 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%cst_22, %cst_22, %cst_22 : tensor<351xi64>, tensor<351xi64>, tensor<351xi64>) outs(%25 : tensor<351xf32>) {
      ^bb0(%in: i64, %in_109: i64, %in_110: i64, %out: f32):
        %c0 = arith.constant 0 : index
        %extracted = tensor.extract %expanded_87[%c0, %c0, %c0] : tensor<1x27x29xf32>
        linalg.yield %extracted : f32
      } -> tensor<351xf32>
      %expanded_95 = tensor.expand_shape %26 [[0, 1]] output_shape [1, 351] : tensor<351xf32> into tensor<1x351xf32>
      %slice0 = tensor.extract_slice %input0[0, 0] [1, 1] [1, 1] : tensor<1x13xf32> to tensor<1x1xf32>
      %res_slice = tensor.extract_slice %expanded_95[0, 0] [1, 1] [1, 1] : tensor<1x351xf32> to tensor<1x1xf32>
      %final_res = arith.addf %slice0, %res_slice : tensor<1x1xf32>
      secret.yield %final_res : tensor<1x1xf32>
    } -> !secret.secret<tensor<1x1xf32>>
    return %0 : !secret.secret<tensor<1x1xf32>>
  }
}
