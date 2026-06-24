// RUN: heir-opt %s --rotom-materialize-tensor-ext-layout | FileCheck %s

// A tensor axis split mixed-radix: dim id 0 appears as two pieces ([0:2:2] and
// [0:2:1]) that together form axis 0 (extent 4), and likewise dim id 1. The
// strides are the within-axis cumulative products (1, then 2), so the two pieces
// share one domain variable. The cumulative product of extents crosses n = 8
// after the slot pieces, putting the high part of axis 0 on the ciphertext axis.
#d0s2 = #rotom.dim<[0:2:2]>
#d1s2 = #rotom.dim<[1:2:2]>
#d0s1 = #rotom.dim<[0:2:1]>
#d1s1 = #rotom.dim<[1:2:1]>
#layout = #rotom.layout<n = 8, dims = [#d0s2, #d1s2, #d0s1, #d1s1]>

// CHECK:   func.func @f(%arg0: tensor<4x4xf32> {tensor_ext.layout =
// CHECK-DAG: #tensor_ext.layout<
// CHECK-DAG: [i0, i1] -> [ct, slot] :
// CHECK-DAG: ct = floor((i0) / 2)
// CHECK-DAG: slot = 2 * (i0 - 2 * floor((i0) / 2)) + 4 * floor((i1) / 2) + (i1 - 2 * floor((i1) / 2))
module {
  func.func @f(%arg0: tensor<4x4xf32> {rotom.layout = #layout}) {
    return
  }
}
