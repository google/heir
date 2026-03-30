// RUN: heir-opt %s --rotom-materialize-tensor-ext-layout | FileCheck %s

// Tiled row-major style: dim ids repeat across traversals, but (dim, stride)
// pairs are distinct. With ``n = 8``, Rotom ``;`` split: first traversal in
// ``dims`` is ciphertext index, the rest pack within each CT.
#d0s2 = #rotom.dim<dim = 0, size = 2, stride = 2>
#d1s2 = #rotom.dim<dim = 1, size = 2, stride = 2>
#d0s1 = #rotom.dim<dim = 0, size = 2, stride = 1>
#d1s1 = #rotom.dim<dim = 1, size = 2, stride = 1>
#layout = #rotom.layout<dims = [#d0s2, #d1s2, #d0s1, #d1s1], n = 8>

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
