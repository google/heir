// RUN: heir-opt %s --rotom-materialize-tensor-ext-layout | FileCheck %s

// Rotom ``[0:4:1][R:2:1]`` with ``n = 8``: row-major where each value is
// repeated twice.
#d0 = #rotom.dim<[0:4:1]>
#r0 = #rotom.dim<[-1:2:4]>
#layout = #rotom.layout<n = 8, dims = [#d0, #r0]>

// CHECK:   func.func @f(%arg0: tensor<4xf32> {tensor_ext.layout =
// CHECK-DAG: #tensor_ext.layout<
// CHECK-DAG: [i0] -> [ct, slot] :
// CHECK-DAG: exists d1
// CHECK-DAG: ct = 0
// CHECK-DAG: slot = 2 * i0 + d1
module {
  func.func @f(%arg0: tensor<4xf32> {rotom.layout = #layout}) {
    return
  }
}
