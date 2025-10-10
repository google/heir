// RUN: heir-opt %s --split-input-file --convert-to-ciphertext-semantics=ciphertext-size=32 | FileCheck %s


// Test that a vector of size 16xi16 is replicated to 1x32xi16.
// CHECK: @repeat_vector
#new_layout = #tensor_ext.new_layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 16 = 0 and 0 <= i0 <= 15 and 0 <= slot <= 31 }">
module {
  func.func @repeat_vector() {
    // CHECK-DAG: %[[c1_i16:.*]] = arith.constant 1 : i16
    // CHECK-DAG: %[[c32:.*]] = arith.constant 32 : index
    // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
    %cst = arith.constant dense<1> : tensor<16xi16>
    // CHECK: %[[cst:.*]] = arith.constant dense<0> : tensor<1x32xi16>
    // CHECK: scf.for %[[arg0:.*]] = %[[c0]] to %[[c32]] step %[[c1]]
    // CHECK: tensor.insert %[[c1_i16]]
    %0 = secret.generic() {
      %1 = tensor_ext.assign_layout %cst {layout = #new_layout, tensor_ext.layout = #new_layout} : tensor<16xi16>
      secret.yield %1 : tensor<16xi16>
    } -> (!secret.secret<tensor<16xi16>> {tensor_ext.layout = #new_layout})
    return
  }
}

// -----

#new_layout = #tensor_ext.new_layout<"{ [] -> [ct, slot] : ct = 0 and 0 <= slot <= 31 }">
module {
  // CHECK: @scalar_mul
  func.func @scalar_mul(%arg0: !secret.secret<i16> {tensor_ext.layout = #new_layout}) -> (!secret.secret<i16> {tensor_ext.layout = #new_layout}) {
    // Dropping unit dims results in a 32xi16 tensor for assign layout
    // CHECK: %[[cst:.*]] = arith.constant dense<2> : tensor<1x32xi16>
    %0 = secret.generic(%arg0: !secret.secret<i16> {tensor_ext.layout = #new_layout}) {
    ^body(%input0: i16):
      %c2_i16 = arith.constant 2 : i16
      %1 = tensor_ext.assign_layout %c2_i16 {layout = #new_layout, tensor_ext.layout = #new_layout} : i16
      %2 = arith.muli %input0, %1 {tensor_ext.layout = #new_layout} : i16
      secret.yield %2 : i16
    } -> (!secret.secret<i16> {tensor_ext.layout = #new_layout})
    // CHECK: return {{.*}} : !secret.secret<tensor<1x32xi16>>
    return %0 : !secret.secret<i16>
  }
}
