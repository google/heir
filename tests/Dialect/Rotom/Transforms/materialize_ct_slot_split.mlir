// RUN: heir-opt %s --rotom-materialize-tensor-ext-layout | FileCheck %s

// Rotom ``[0:2:2][1:2:2][0:2:1][1:2:1]`` with ``n = 8`` on a 4x4 tensor: each
// tensor axis is split mixed-radix into a stride-2 (high) and stride-1 (low)
// piece. The cumulative product of extents crosses n after the slot pieces, so
// the high part of axis 0 indexes ciphertexts. This is a 2x2-tiled row-major
// layout (ct = tile-row, slot = 4*tile-col + 2*within-row + within-col).
#d0 = #rotom.dim<[0:2:2]>
#d1 = #rotom.dim<[1:2:2]>
#d2 = #rotom.dim<[0:2:1]>
#d3 = #rotom.dim<[1:2:1]>
#layout = #rotom.layout<n = 8, dims = [#d0, #d1, #d2, #d3]>

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
