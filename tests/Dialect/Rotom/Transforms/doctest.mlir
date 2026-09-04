// RUN: heir-opt %s --rotom-materialize-tensor-ext-layout | FileCheck %s

// Rotom layouts attach to SSA values the same way as `tensor_ext.layout` in
// other pipelines (e.g. convert-to-ciphertext-semantics tests): function
// arguments, region arguments, and op results via the attribute-association
// rules used by `findAttributeAssociatedWith`.

#d0 = #rotom.dim<[0:4:1]>
#d1 = #rotom.dim<[1:4:1]>
#layout = #rotom.layout<n = 16, dims = [#d0, #d1]>

// A layouted CLEARTEXT producer (here the constant) is an encode-time packing
// boundary: it keeps no layout of its own and its value is routed through an
// explicit tensor_ext.assign_layout instead, so the producer chain stays
// cleartext and the packing has a single lowerable form.
// CHECK:   func.func @f(%arg0: tensor<4x4xf32> {tensor_ext.layout =
// CHECK: arith.constant
// CHECK-NEXT: tensor_ext.assign_layout
// CHECK-SAME: layout =
// CHECK-NOT: rotom.layout
module {
  func.func @f(%arg0: tensor<4x4xf32> {rotom.layout = #layout}) -> tensor<4x4xf32> {
    %c = arith.constant {rotom.layout = #layout} dense<0.0> : tensor<4x4xf32>
    return %c : tensor<4x4xf32>
  }
}
