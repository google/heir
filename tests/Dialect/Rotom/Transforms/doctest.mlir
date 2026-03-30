// RUN: heir-opt %s --rotom-materialize-tensor-ext-layout | FileCheck %s

// Rotom layouts attach to SSA values the same way as `tensor_ext.layout` in
// other pipelines (e.g. convert-to-ciphertext-semantics tests): function
// arguments, region arguments, and op results via the attribute-association
// rules used by `findAttributeAssociatedWith`.

#d0 = #rotom.dim<dim = 0, size = 4, stride = 1>
#d1 = #rotom.dim<dim = 1, size = 4, stride = 1>
#layout = #rotom.layout<dims = [#d0, #d1], n = 16>

// CHECK:   func.func @f(%arg0: tensor<4x4xf32> {tensor_ext.layout =
// CHECK: arith.constant {tensor_ext.layout =
// CHECK-NOT: rotom.layout
module {
  func.func @f(%arg0: tensor<4x4xf32> {rotom.layout = #layout}) -> tensor<4x4xf32> {
    %c = arith.constant {rotom.layout = #layout} dense<0.0> : tensor<4x4xf32>
    return %c : tensor<4x4xf32>
  }
}
