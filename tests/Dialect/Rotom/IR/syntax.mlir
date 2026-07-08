// RUN: heir-opt %s | FileCheck %s

#d0 = #rotom.dim<[0:4:1]>
#d1 = #rotom.dim<[1:4:1]>
// Unused slot capacity is spelled as an explicit gap piece (the builders
// insert it; written layouts must show it).
#plain = #rotom.layout<n = 8, dims = [[G:2:1], [0:4:1]]>
#rolled = #rotom.layout<n = 16, rolls = [(0, 1)], dims = [#d0, #d1]>
// Replication and gap dims print as R and G; the numeric ids -1 and -2 are
// accepted on input and round-trip to the letter forms.
#repl_gap = #rotom.layout<n = 16, dims = [[R:2:1], [-2:2:1], [0:4:1]]>
// The `|` separates ciphertext dims from slot dims (omitted when every dim
// is a slot dim): both k digits index ciphertexts here, i addresses slots.
// A roll endpoint is a dims-list position (bare integer, one piece) or a
// whole tensor axis (`axis N`, legal only when the axis is split): the axis
// FROM rewrites the whole split-k index and each piece takes its digit.
// Rolls always shift by exactly the partner index; kernel-schedule shifts
// (e.g. the BSGS giant pre-rotation) are plan-level encodings, not layout
// vocabulary.
#axis = #rotom.layout<n = 16, rolls = [(axis 1, 2)], dims = [[1:4:4], [1:4:1] | [0:16:1]]>
// Rolls may be written flat (bare integer pairs) or as tuples, and the two
// forms may be mixed; each entry is one pair. Everything round-trips to the
// tuple form.
#mixed = #rotom.layout<n = 16, rolls = [0, 1, (2, 3)], dims = [[0:2:1], [1:2:1], [2:2:1], [3:2:1]]>
// An oversized dim spans ciphertexts entirely (ct = i0): here each of the 8
// elements sits in its own ciphertext, and the slot side is all gap.
#split = #rotom.layout<n = 4, dims = [[0:8:1] | [G:4:1]]>

// CHECK: #dim = #rotom.dim<[2:8:4]>
// CHECK: #layout = #rotom.layout<n = 16, rolls = [(axis 1, 2)], dims = {{\[\[1:4:4\], \[1:4:1\] \| \[0:16:1\]\]}}>
// CHECK: #layout1 = #rotom.layout<n = 8, dims = {{\[\[G:2:1\], \[0:4:1\]\]}}>
// CHECK: #layout2 = #rotom.layout<n = 16, dims = {{\[\[R:2:1\], \[G:2:1\], \[0:4:1\]\]}}>
// CHECK: #layout3 = #rotom.layout<n = 16, rolls = [(0, 1)], dims = {{\[\[0:4:1\], \[1:4:1\]\]}}>
// CHECK: #layout4 = #rotom.layout<n = 4, dims = {{\[\[0:8:1\] \| \[G:4:1\]\]}}>
// CHECK: #layout5 = #rotom.layout<n = 16, rolls = [(0, 1), (2, 3)], dims = {{\[\[0:2:1\], \[1:2:1\], \[2:2:1\], \[3:2:1\]\]}}>
// CHECK: module attributes
// CHECK-SAME: rotom.axis_layout = #layout
// CHECK-SAME: rotom.dim_attr = #dim
// CHECK-SAME: rotom.plain_layout = #layout1
// CHECK-SAME: rotom.repl_gap_layout = #layout2
// CHECK-SAME: rotom.rolled_layout = #layout3
// CHECK-SAME: rotom.split_layout = #layout4
// CHECK-SAME: rotom.zmixed_layout = #layout5
module attributes {
  rotom.dim_attr = #rotom.dim<[2:8:4]>,
  rotom.plain_layout = #plain,
  rotom.repl_gap_layout = #repl_gap,
  rotom.rolled_layout = #rolled,
  rotom.axis_layout = #axis,
  rotom.split_layout = #split,
  rotom.zmixed_layout = #mixed
} {
  func.func @f(%arg0: tensor<4x4xf32>) {
    return
  }
}
