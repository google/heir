
// Tests use-submodules error when optimizing a generic distributed through
// affine for loops.

// RUN: heir-opt -yosys-optimizer="abc-fast=true use-submodules=true" %s --verify-diagnostics

module attributes {tf_saved_model.semantics} {
  func.func @main(%arg0: !secret.secret<tensor<1x1xi16>>, %0: !secret.secret<tensor<1x1xi8>>) -> (!secret.secret<tensor<1x1xi8>>) {
    %c127_i32 = arith.constant 127 : i16
    %out = affine.for %arg1 = 0 to 1 iter_args(%outerarg = %0) -> (!secret.secret<tensor<1x1xi8>>) {
      %4 = affine.for %arg2 = 0 to 1 iter_args(%innerarg = %outerarg) -> (!secret.secret<tensor<1x1xi8>>) {
        %1 = secret.generic(%arg0: !secret.secret<tensor<1x1xi16>>, %arg1: index, %arg2: index) {
        ^bb0(%arg3: tensor<1x1xi16>, %arg4: index, %arg5: index):
          %3 = tensor.extract %arg3[%arg4, %arg5] : tensor<1x1xi16>
          secret.yield %3 : i16
        } -> !secret.secret<i16>
        %2 = secret.generic(%1: !secret.secret<i16>, %c127_i32: i16) {
        ^bb0(%arg3: i16, %arg4: i16):
          %3 = arith.addi %arg3, %arg4 : i16
          %4 = arith.trunci %3 : i16 to i8
          secret.yield %4 : i8
        } -> !secret.secret<i8>
        %3 = secret.generic(%innerarg: !secret.secret<tensor<1x1xi8>>, %2:!secret.secret<i8>, %arg1: index, %arg2 : index) {
        ^bb0(%arg3: tensor<1x1xi8>, %arg4: i8, %arg5: index, %arg6: index):
          %inserted = tensor.insert %arg4 into %arg3[%arg5, %arg6] : tensor<1x1xi8>
          secret.yield %inserted : tensor<1x1xi8>
        } -> !secret.secret<tensor<1x1xi8>>
        affine.yield %3 : !secret.secret<tensor<1x1xi8>>
      }
      affine.yield %4 : !secret.secret<tensor<1x1xi8>>
    }
    return %out : !secret.secret<tensor<1x1xi8>>
  }
}
