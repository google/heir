// RUN: heir-opt --tosa-to-boolean-tfhe=abc-fast=true %s | FileCheck %s

// CHECK-LABEL: module
module attributes {tf_saved_model.semantics} {
  // CHECK: @main([[sks:.*]]: !tfhe_rust.server_key, [[arg:.*]]: memref<1x1x8x!tfhe_rust.eui3>) -> memref<1x1x32x!tfhe_rust.eui3>
  func.func @main(%11: tensor<1x1xi8>) -> tensor<1x1xi32> {
    %0 = "tosa.const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
    %1 = "tosa.const"() {value = dense<[[1]]> : tensor<1x1xi8>} : () -> tensor<1x1xi8>
    %12 = "tosa.fully_connected"(%11, %1, %0) {input_zp = -128 : i32, weight_zp = 0 : i32} : (tensor<1x1xi8>, tensor<1x1xi8>, tensor<1xi32>) -> tensor<1x1xi32>
    // CHECK: [[ALLOC:%.*]] = memref.alloc()
    // CHECK-SAME: memref<1x1x32x!tfhe_rust.eui3>
    // CHECK-NOT: comb
    // CHECK-NOT: memref.subview
    // CHECK-NOT: affine.load
    // CEHCK-NOT: affine.store
    // CHECK: return [[ALLOC]] : memref<1x1x32x!tfhe_rust.eui3>
    return %12 : tensor<1x1xi32>
  }
}
