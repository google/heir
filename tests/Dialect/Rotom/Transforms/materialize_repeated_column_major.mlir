// RUN: heir-opt %s --rotom-materialize-tensor-ext-layout | FileCheck %s

// Rotom ``[R:4:1];[1:4:1][0:4:1]`` with ``n = 16`` (replication + column-major
// traversals in ``dims``). Replication is projected to ciphertext index ``ct``
// via existential ``d2``; slots pack ``i0 + 4 * i1``.
#d0 = #rotom.dim<dim = 0, size = 4, stride = 1>
#d1 = #rotom.dim<dim = 1, size = 4, stride = 1>
#d2 = #rotom.dim<dim = -1, size = 4, stride = 1>
#layout = #rotom.layout<dims = [#d2, #d1, #d0], n = 16>

// CHECK:   func.func @f(%arg0: tensor<4x4xf32> {tensor_ext.layout =
// CHECK-DAG: #tensor_ext.layout<
// CHECK-DAG: [i0, i1] -> [ct, slot] :
// CHECK-DAG: exists d2
// CHECK-DAG: ct = d2
// CHECK-DAG: slot = i0 + 4 * i1
module {
  func.func @f(%arg0: tensor<4x4xf32> {rotom.layout = #layout}) {
    return
  }
}
