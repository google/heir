// RUN: heir-opt %s | FileCheck %s

#d0 = #rotom.dim<[0:4:1]>
#d1 = #rotom.dim<[1:4:1]>
#plain = #rotom.layout<n = 8, dims = [[0:4:1]]>
#rolled = #rotom.layout<n = 16, rolls = [(0, 1)], dims = [#d0, #d1]>

// CHECK: #dim = #rotom.dim<[2:8:4]>
// CHECK: #layout = #rotom.layout<n = 8, dims = {{\[\[0:4:1\]\]}}>
// CHECK: #layout1 = #rotom.layout<n = 16, rolls = [(0, 1)], dims = {{\[\[0:4:1\], \[1:4:1\]\]}}>
// CHECK: module attributes
// CHECK-SAME: rotom.dim_attr = #dim
// CHECK-SAME: rotom.plain_layout = #layout
// CHECK-SAME: rotom.rolled_layout = #layout1
module attributes {
  rotom.dim_attr = #rotom.dim<[2:8:4]>,
  rotom.plain_layout = #plain,
  rotom.rolled_layout = #rolled
} {
  func.func @f(%arg0: tensor<4x4xf32>) {
    return
  }
}
