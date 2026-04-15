// RUN: heir-opt %s --rotom-materialize-tensor-ext-layout | FileCheck %s

// Rotom ``[0:4:1][G:2:1]`` with ``n = 8``: in-slot row with explicit gap.
#d0 = #rotom.dim<dim = 0, size = 4, stride = 1>
#g0 = #rotom.dim<dim = -2, size = 2, stride = 1>
#layout = #rotom.layout<dims = [#d0, #g0], n = 8>

// CHECK:   func.func @f(%arg0: tensor<4xf32> {tensor_ext.layout =
// CHECK-DAG: #tensor_ext.layout<
// CHECK-DAG: [i0] -> [ct, slot] :
// CHECK-DAG: ct = 0
// CHECK-DAG: slot = 2 * i0
module {
  func.func @f(%arg0: tensor<4xf32> {rotom.layout = #layout}) {
    return
  }
}
