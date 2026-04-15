// RUN: heir-opt %s --rotom-materialize-tensor-ext-layout | FileCheck %s

// Rotom ``[0:4:1][1:4:1]`` with ``n = 16``: row-major.
#d0 = #rotom.dim<dim = 0, size = 4, stride = 1>
#d1 = #rotom.dim<dim = 1, size = 4, stride = 1>
#layout = #rotom.layout<dims = [#d0, #d1], n = 16>

// CHECK:   func.func @f(%arg0: tensor<4x4xf32> {tensor_ext.layout =
// CHECK-DAG: #tensor_ext.layout<
// CHECK-DAG: [i0, i1] -> [ct, slot] :
// CHECK-DAG: ct = 0
// CHECK-DAG: slot = 4 * i0 + i1
module {
  func.func @f(%arg0: tensor<4x4xf32> {rotom.layout = #layout}) {
    return
  }
}
