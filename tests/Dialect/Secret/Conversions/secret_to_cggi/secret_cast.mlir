// This test ensures that secret casting before and after generics lowers to CGGI properly.

// RUN: heir-opt --secret-distribute-generic --secret-to-cggi -cse --split-input-file %s | FileCheck %s

// CHECK: module
module attributes {tf_saved_model.semantics} {
  // CHECK: @main([[ARG:%.*]]: [[LWET:tensor<4x!lwe.lwe_ciphertext<.*>>]]) -> [[OUT:tensor<4x!lwe.lwe_ciphertext<.*>>]]
  func.func @main(%arg0: !secret.secret<tensor<4xi1>>) -> !secret.secret<i4> {
    // CHECK: return [[ARG]] : [[OUT]]
    %4 = secret.cast %arg0 : !secret.secret<tensor<4xi1>> to !secret.secret<i4>
    return %4 : !secret.secret<i4>
  }
}

// -----

// These casts occur when converting op operands and results from yosys optimized blocks.

module {
  // CHECK-NOT: secret
  // CHECK: @collapse_multidim([[ARG:%.*]]: tensor<1x1x8x[[LWET:!lwe.lwe_ciphertext<.*>]]>)
  func.func @collapse_multidim(%arg0: !secret.secret<tensor<1x1xi8>>) -> !secret.secret<tensor<8xi1>> {
    // CHECK: tensor.collapse_shape
    // CHECK-SAME: tensor<1x1x8x[[LWET]]> into tensor<8x[[LWET]]>
    %0 = secret.cast %arg0 : !secret.secret<tensor<1x1xi8>> to !secret.secret<tensor<8xi1>>
    func.return %0 : !secret.secret<tensor<8xi1>>
  }

  // CHECK: @expand_multidim([[ARG:%.*]]: tensor<16x[[LWET:!lwe.lwe_ciphertext<.*>]]>)
  func.func @expand_multidim(%arg0: !secret.secret<tensor<16xi1>>) -> !secret.secret<tensor<1x2xi8>> {
    // CHECK: tensor.expand_shape
    // CHECK-SAME: tensor<16x[[LWET]]> into tensor<1x2x8x[[LWET]]>
    %3 = secret.cast %arg0 : !secret.secret<tensor<16xi1>> to !secret.secret<tensor<1x2xi8>>
    return %3 : !secret.secret<tensor<1x2xi8>>
  }
}
