// RUN: heir-opt %s | FileCheck %s

#d0 = #rotom.dim<[0:4:1]>
#d1 = #rotom.dim<[1:4:1]>
#plain = #rotom.layout<n = 8, dims = [[0:4:1]]>
#rolled = #rotom.layout<n = 16, rolls = [(0, 1)], dims = [#d0, #d1]>
// Replication and gap dims print as R and G; the numeric ids -1 and -2 are
// accepted on input and round-trip to the letter forms.
#repl_gap = #rotom.layout<n = 16, dims = [[R:2:1], [-2:2:1], [0:4:1]]>

// CHECK: #dim = #rotom.dim<[2:8:4]>
// CHECK: #layout = #rotom.layout<n = 8, dims = {{\[\[0:4:1\]\]}}>
// CHECK: #layout1 = #rotom.layout<n = 16, dims = {{\[\[R:2:1\], \[G:2:1\], \[0:4:1\]\]}}>
// CHECK: #layout2 = #rotom.layout<n = 16, rolls = [(0, 1)], dims = {{\[\[0:4:1\], \[1:4:1\]\]}}>
// CHECK: module attributes
// CHECK-SAME: rotom.dim_attr = #dim
// CHECK-SAME: rotom.plain_layout = #layout
// CHECK-SAME: rotom.repl_gap_layout = #layout1
// CHECK-SAME: rotom.rolled_layout = #layout2
module attributes {
  rotom.dim_attr = #rotom.dim<[2:8:4]>,
  rotom.plain_layout = #plain,
  rotom.repl_gap_layout = #repl_gap,
  rotom.rolled_layout = #rolled
} {
  func.func @f(%arg0: tensor<4x4xf32>) {
    return
  }
}
