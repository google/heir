// RUN: heir-opt --canonicalize %s | FileCheck %s

// Regression test to assure HoistPlaintextOps does not
// mistakenly extract complex op before secret.generic
// which may cause assertion fail in RemoveUnusedGenericArgs

// CHECK-LABEL: @simple_sum_with_thick_body
func.func @simple_sum_with_thick_body(%arg0: !secret.secret<tensor<32xi16>>) -> !secret.secret<i16> {
  %c0_i16 = arith.constant 0 : i16
  %0 = secret.generic ins(%arg0 : !secret.secret<tensor<32xi16>>) {
  ^bb0(%arg1: tensor<32xi16>):
    // assure body is thick enough for HoistPlaintextOps to happen
    %1 = arith.addi %arg1, %arg1 : tensor<32xi16>
    %2 = arith.addi %1, %arg1 : tensor<32xi16>
    %3 = arith.addi %2, %arg1 : tensor<32xi16>
    %4 = affine.for %arg2 = 0 to 32 iter_args(%arg3 = %c0_i16) -> (i16) {
      %extracted = tensor.extract %3[%arg2] : tensor<32xi16>
      %5 = arith.addi %extracted, %arg3 : i16
      affine.yield %5 : i16
    }
    secret.yield %4 : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}

// CHECK-LABEL: @simple_sum_tiled_with_thick_body
func.func @simple_sum_tiled_with_thick_body(%arg0: !secret.secret<tensor<32xi16>>) -> !secret.secret<i16> {
  %c0_i16 = arith.constant 0 : i16
  %c8 = arith.constant 8 : index
  %0 = secret.generic ins(%arg0 : !secret.secret<tensor<32xi16>>) {
  ^bb0(%arg1: tensor<32xi16>):
    // assure body is thick enough for HoistPlaintextOps to happen
    %1 = arith.addi %arg1, %arg1 : tensor<32xi16>
    %2 = arith.addi %1, %arg1 : tensor<32xi16>
    %3 = arith.addi %2, %arg1 : tensor<32xi16>
    %4 = affine.for %arg2 = 0 to 4 iter_args(%arg3 = %c0_i16) -> (i16) {
      %5 = affine.for %arg4 = 0 to 8 iter_args(%arg5 = %c0_i16) -> (i16) {
        %7 = arith.muli %arg2, %c8 : index
        %8 = arith.addi %7, %arg4 : index
        %extracted = tensor.extract %3[%8] : tensor<32xi16>
        %9 = arith.addi %extracted, %arg5 : i16
        affine.yield %9 : i16
      }
      %6 = arith.addi %5, %arg3 : i16
      affine.yield %6 : i16
    }
    secret.yield %4 : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}
