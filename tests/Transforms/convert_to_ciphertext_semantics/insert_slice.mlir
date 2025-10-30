// RUN: heir-opt --convert-to-ciphertext-semantics=ciphertext-size=32 --split-input-file %s | FileCheck %s

#layout = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and ct = i1 and (-4i2 - i3 + slot) mod 16 = 0 and 0 <= i1 <= 1 and 0 <= i2 <= 3 and 0 <= i3 <= 3 and 0 <= slot <= 31 }">
#layout1 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-4i0 - i1 + slot) mod 16 = 0 and 0 <= i0 <= 3 and 0 <= i1 <= 3 and 0 <= slot <= 31 }">
module {
  // Layouts are aligned perfectly so that %input0 is directly inserted into the destination as a single ciphertext.
  // CHECK: func.func @trivial_insert
  func.func @trivial_insert(%arg0: !secret.secret<tensor<4x4xf32>> {tensor_ext.layout = #layout1}) -> (!secret.secret<tensor<1x2x4x4xf32>> {tensor_ext.layout = #layout}) {
    %0 = tensor.empty() : tensor<1x2x4x4xf32>
    // CHECK: %[[v0:.*]] = tensor.empty() : tensor<1x2x4x4xf32>
    // CHECK: secret.generic
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: %[[v3:.*]] = tensor.empty() : tensor<2x32xf32>
    // CHECK: arith.addf
    // CHECK-COUNT-2: tensor.insert_slice
    // CHECK-COUNT-2: tensor.insert_slice
    // CHECK: arith.addf
    // CHECK: return
    %1 = secret.generic(%arg0: !secret.secret<tensor<4x4xf32>> {tensor_ext.layout = #layout1}) {
    ^body(%input0: tensor<4x4xf32>):
      %2 = tensor_ext.assign_layout %0 {layout = #layout, tensor_ext.layout = #layout} : tensor<1x2x4x4xf32>
      %inserted_slice = tensor.insert_slice %input0 into %2[0, 0, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] {tensor_ext.layout = #layout} : tensor<4x4xf32> into tensor<1x2x4x4xf32>
      %inserted_slice_0 = tensor.insert_slice %input0 into %inserted_slice[0, 1, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] {tensor_ext.layout = #layout} : tensor<4x4xf32> into tensor<1x2x4x4xf32>
      %3 = arith.addf %inserted_slice_0, %inserted_slice_0 {tensor_ext.layout = #layout} : tensor<1x2x4x4xf32>
      secret.yield %3 : tensor<1x2x4x4xf32>
    } -> (!secret.secret<tensor<1x2x4x4xf32>> {tensor_ext.layout = #layout})
    return %1 : !secret.secret<tensor<1x2x4x4xf32>>
  }
}

// -----

#layout = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and ct = i1 and (-4i2 - i3 + slot) mod 16 = 0 and 0 <= i1 <= 1 and 0 <= i2 <= 3 and 0 <= i3 <= 3 and 0 <= slot <= 31 }">
#layout1 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-4i1 - i0 + slot) mod 16 = 0 and 0 <= i0 <= 3 and 0 <= i1 <= 3 and 0 <= slot <= 31 }">
module {
  // The input must be remapped to the destination layout before inserting.
  // CHECK: func.func @remap_input
  func.func @remap_input(%arg0: !secret.secret<tensor<4x4xf32>> {tensor_ext.layout = #layout1}) -> (!secret.secret<tensor<1x2x4x4xf32>> {tensor_ext.layout = #layout}) {
    %0 = tensor.empty() : tensor<1x2x4x4xf32>
    // CHECK: %[[v0:.*]] = tensor.empty() : tensor<1x2x4x4xf32>
    // CHECK: secret.generic
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: tensor_ext.remap
    // CHECK: %[[v3:.*]] = tensor.empty() : tensor<2x32xf32>
    // CHECK: arith.addf
    // CHECK-COUNT-2: tensor.insert_slice
    // CHECK: arith.addf
    // CHECK: return
    %1 = secret.generic(%arg0: !secret.secret<tensor<4x4xf32>> {tensor_ext.layout = #layout1}) {
    ^body(%input0: tensor<4x4xf32>):
      %2 = tensor_ext.assign_layout %0 {layout = #layout, tensor_ext.layout = #layout} : tensor<1x2x4x4xf32>
      %inserted_slice = tensor.insert_slice %input0 into %2[0, 0, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] {tensor_ext.layout = #layout} : tensor<4x4xf32> into tensor<1x2x4x4xf32>
      %3 = arith.addf %inserted_slice, %inserted_slice {tensor_ext.layout = #layout} : tensor<1x2x4x4xf32>
      secret.yield %3 : tensor<1x2x4x4xf32>
    } -> (!secret.secret<tensor<1x2x4x4xf32>> {tensor_ext.layout = #layout})
    return %1 : !secret.secret<tensor<1x2x4x4xf32>>
  }
}

// -----

// Ensure we don't need to remap the source tensor even if we insert a slice at a non-zero offset.

#layout = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and ct = i1 and (-4i2 - i3 + slot) mod 16 = 0 and 0 <= i1 <= 1 and 0 <= i2 <= 3 and 0 <= i3 <= 3 and 0 <= slot <= 31 }">
#layout1 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-4i0 - i1 + slot) mod 16 = 0 and 0 <= i0 <= 3 and 0 <= i1 <= 3 and 0 <= slot <= 31 }">
module {
  // CHECK: func.func @offset
  // CHECK-NOT: tensor_ext.remap
  // CHECK: tensor.insert_slice
  // CHECK: return
  func.func @offset(%arg0: !secret.secret<tensor<4x4xf32>> {tensor_ext.layout = #layout1}) -> (!secret.secret<tensor<1x2x4x4xf32>> {tensor_ext.layout = #layout}) {
    %0 = tensor.empty() : tensor<1x2x4x4xf32>
    %1 = secret.generic(%arg0: !secret.secret<tensor<4x4xf32>> {tensor_ext.layout = #layout1}) {
    ^body(%input0: tensor<4x4xf32>):
      %2 = tensor_ext.assign_layout %0 {layout = #layout, tensor_ext.layout = #layout} : tensor<1x2x4x4xf32>
      %inserted_slice = tensor.insert_slice %input0 into %2[0, 1, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] {tensor_ext.layout = #layout} : tensor<4x4xf32> into tensor<1x2x4x4xf32>
      secret.yield %inserted_slice : tensor<1x2x4x4xf32>
    } -> (!secret.secret<tensor<1x2x4x4xf32>> {tensor_ext.layout = #layout})
    return %1 : !secret.secret<tensor<1x2x4x4xf32>>
  }
}
