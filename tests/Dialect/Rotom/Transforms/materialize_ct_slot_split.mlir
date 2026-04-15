// RUN: heir-opt %s --rotom-materialize-tensor-ext-layout | FileCheck %s

// Rotom ``[0:2:2];[1:2:2][0:2:1][1:2:1]`` with ``n = 8``: first dim is
// ciphertext traversal, remaining dims pack within each CT. This is a tiled
// row-major layout.
#d0 = #rotom.dim<dim = 0, size = 2, stride = 2>
#d1 = #rotom.dim<dim = 1, size = 2, stride = 2>
#d2 = #rotom.dim<dim = 0, size = 2, stride = 1>
#d3 = #rotom.dim<dim = 1, size = 2, stride = 1>
#layout = #rotom.layout<dims = [#d0, #d1, #d2, #d3], n = 8>

// CHECK:   func.func @f(%arg0: tensor<2x2x2x2xf32> {tensor_ext.layout =
// CHECK-DAG: #tensor_ext.layout<
// CHECK-DAG: [i0, i1, i2, i3] -> [ct, slot] :
// CHECK-DAG: ct = i0
// CHECK-DAG: slot = 4 * i1 + 2 * i2 + i3
module {
  func.func @f(%arg0: tensor<2x2x2x2xf32> {rotom.layout = #layout}) {
    return
  }
}
