// RUN: heir-opt %s --rotom-materialize-tensor-ext-layout | FileCheck %s

// Rotom ``roll(0, 1)`` on a 4x4 row-major layout: slot indices group
// diagonal ``(i0 - i1) mod 4`` classes in row-major slot order.
#d0 = #rotom.dim<[0:4:1]>
#d1 = #rotom.dim<[1:4:1]>
#layout = #rotom.layout<n = 16, rolls = [(0, 1)], dims = [#d0, #d1]>

// CHECK:   func.func @f(%arg0: tensor<4x4xf32> {tensor_ext.layout =
// CHECK-DAG: #tensor_ext.layout<
// CHECK-DAG: [i0, i1] -> [ct, slot] :
// CHECK-DAG: ct = 0
// CHECK-DAG: slot = 4 * ((i0 - i1) - 4 * floor(((i0 - i1)) / 4)) + i1
module {
  func.func @f(%arg0: tensor<4x4xf32> {rotom.layout = #layout}) {
    return
  }
}
