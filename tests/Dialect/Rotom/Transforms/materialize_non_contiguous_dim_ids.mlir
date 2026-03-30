// RUN: heir-opt %s --rotom-materialize-tensor-ext-layout | FileCheck %s

#d0 = #rotom.dim<dim = 0, size = 2, stride = 1>
#d2 = #rotom.dim<dim = 2, size = 2, stride = 1>
#layout = #rotom.layout<dims = [#d0, #d2], n = 4>

// CHECK:   func.func @f(%arg0: tensor<2x2x2xf32> {tensor_ext.layout =
// CHECK-DAG: #tensor_ext.layout<
// CHECK-DAG: [i0, i1] -> [ct, slot] :
// CHECK-DAG: ct = 0
// CHECK-DAG: slot = 2 * i0 + i1
module {
  func.func @f(%arg0: tensor<2x2x2xf32> {rotom.layout = #layout}) {
    return
  }
}
