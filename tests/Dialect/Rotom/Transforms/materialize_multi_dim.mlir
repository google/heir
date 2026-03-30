// RUN: heir-opt %s --rotom-materialize-tensor-ext-layout | FileCheck %s

#d0 = #rotom.dim<dim = 0, size = 2, stride = 1>
#d1 = #rotom.dim<dim = 1, size = 2, stride = 1>
#d2 = #rotom.dim<dim = 2, size = 4, stride = 1>
#layout = #rotom.layout<dims = [#d0, #d1, #d2], n = 16>

// CHECK:   func.func @f(%arg0: tensor<2x2x4xf32> {tensor_ext.layout =
// CHECK-DAG: #tensor_ext.layout<
// CHECK-DAG: [i0, i1, i2] -> [ct, slot] :
// CHECK-DAG: ct = 0
// CHECK-DAG: slot = 8 * i0 + 4 * i1 + i2
module {
  func.func @f(%arg0: tensor<2x2x4xf32> {rotom.layout = #layout}) {
    return
  }
}
