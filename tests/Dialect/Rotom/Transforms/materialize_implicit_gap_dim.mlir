// RUN: heir-opt %s --rotom-materialize-tensor-ext-layout | FileCheck %s


// Rotom ``[G:2:1][0:4:1]`` with ``n = 8``: Row-major, first 4. Implicit gap
// dimension, ``[G:2:1]``, should be added in front.
#d0 = #rotom.dim<dim = 0, size = 4, stride = 1>
#layout = #rotom.layout<dims = [#d0], n = 8>

// CHECK:   func.func @f(%arg0: tensor<4xf32> {tensor_ext.layout =
// CHECK-DAG: #tensor_ext.layout<
// CHECK-DAG: [i0] -> [ct, slot] :
// CHECK-DAG: ct = 0
// CHECK-DAG: slot = i0
module {
  func.func @f(%arg0: tensor<4xf32> {rotom.layout = #layout}) {
    return
  }
}
