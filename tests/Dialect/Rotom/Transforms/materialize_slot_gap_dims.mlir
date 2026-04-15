// RUN: heir-opt %s --rotom-materialize-tensor-ext-layout | FileCheck %s

// Two tensor axes (2x2) plus explicit gap dims: linear address includes gap
// variables with g_k = 0 (payload only at gap index 0; other indices zero-fill).
#d0 = #rotom.dim<dim = 0, size = 2, stride = 1>
#d1 = #rotom.dim<dim = 1, size = 2, stride = 2>
#g0 = #rotom.dim<dim = -2, size = 2, stride = 1>
#g1 = #rotom.dim<dim = -2, size = 2, stride = 4>
#layout = #rotom.layout<dims = [#d0, #d1, #g0, #g1], n = 16>

// CHECK:   func.func @f(%arg0: tensor<2x2xf32> {tensor_ext.layout =
// CHECK-DAG: #tensor_ext.layout<
// CHECK-DAG: [i0, i1] -> [ct, slot] :
// CHECK-DAG: ct = 0
// CHECK-DAG: slot = 8 * i0 + 4 * i1
module {
  func.func @f(%arg0: tensor<2x2xf32> {rotom.layout = #layout}) {
    return
  }
}
