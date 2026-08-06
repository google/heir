// RUN: heir-opt %s --convert-to-ciphertext-semantics=min-slot-count=32 --split-input-file | FileCheck %s

// Assigning a layout to a single-element tensor whose layout is dense in the
// ciphertext is a broadcast of its lone element: extract the element once and
// splat it, rather than emitting one loop iteration per slot whose body is
// loop-invariant.
#dense_layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and i0 = 0 and 0 <= slot <= 31 }">

// CHECK: @assign_layout_single_element_dense
module {
  func.func @assign_layout_single_element_dense(%arg0: tensor<1xi16>) -> (!secret.secret<tensor<1xi16>> {tensor_ext.layout = #dense_layout}) {
    // CHECK-NOT: scf.for
    // CHECK-NOT: tensor.insert
    // CHECK: %[[ELT:.*]] = tensor.extract
    // CHECK: tensor.splat %[[ELT]] : tensor<1x32xi16>
    // CHECK-NOT: scf.for
    // CHECK-NOT: tensor.insert
    %0 = secret.generic() {
      %1 = tensor_ext.assign_layout %arg0 {layout = #dense_layout, tensor_ext.layout = #dense_layout} : tensor<1xi16>
      secret.yield %1 : tensor<1xi16>
    } -> (!secret.secret<tensor<1xi16>> {tensor_ext.layout = #dense_layout})
    // CHECK: return
    return %0 : !secret.secret<tensor<1xi16>>
  }
}

// -----

// When the layout is not dense in the ciphertext, the lone element does not
// occupy every slot, so the broadcast above would be wrong. This falls back to
// the general loop-generator path.
#sparse_layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and i0 = 0 and 0 <= slot <= 15 }">

// CHECK: @assign_layout_single_element_not_dense
module {
  func.func @assign_layout_single_element_not_dense(%arg0: tensor<1xi16>) -> (!secret.secret<tensor<1xi16>> {tensor_ext.layout = #sparse_layout}) {
    // CHECK-NOT: tensor.splat
    %0 = secret.generic() {
      %1 = tensor_ext.assign_layout %arg0 {layout = #sparse_layout, tensor_ext.layout = #sparse_layout} : tensor<1xi16>
      secret.yield %1 : tensor<1xi16>
    } -> (!secret.secret<tensor<1xi16>> {tensor_ext.layout = #sparse_layout})
    // CHECK: return
    return %0 : !secret.secret<tensor<1xi16>>
  }
}
