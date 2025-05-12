
// Tests use-submodules error when optimizing a generic distributed through
// affine for loops.

// RUN: heir-opt -yosys-optimizer="abc-fast=true use-submodules=true" %s --verify-diagnostics

module attributes {tf_saved_model.semantics} {
  func.func @main(%arg0: !secret.secret<memref<1x1xi16>>) -> (!secret.secret<memref<1x1xi8>>) {
    %c127_i32 = arith.constant 127 : i16
    %0 = secret.generic() {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1xi8>
      secret.yield %alloc : memref<1x1xi8>
    } -> !secret.secret<memref<1x1xi8>>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        %1 = secret.generic(%arg0: !secret.secret<memref<1x1xi16>>, %arg1: index, %arg2: index) {
        ^bb0(%arg3: memref<1x1xi16>, %arg4: index, %arg5: index):
          %3 = memref.load %arg3[%arg4, %arg5] : memref<1x1xi16>
          secret.yield %3 : i16
        } -> !secret.secret<i16>
        %2 = secret.generic(%1: !secret.secret<i16>, %c127_i32: i16) {
        ^bb0(%arg3: i16, %arg4: i16):
          %3 = arith.addi %arg3, %arg4 : i16
          %4 = arith.trunci %3 : i16 to i8
          secret.yield %4 : i8
        } -> !secret.secret<i8>
        secret.generic(%0: !secret.secret<memref<1x1xi8>>, %2:!secret.secret<i8>, %arg1: index, %arg2 : index) {
        ^bb0(%arg3: memref<1x1xi8>, %arg4: i8, %arg5: index, %arg6: index):
          memref.store %arg4, %arg3[%arg5, %arg6] : memref<1x1xi8>
          secret.yield
        }
      }
    }
    return %0 : !secret.secret<memref<1x1xi8>>
  }
}
