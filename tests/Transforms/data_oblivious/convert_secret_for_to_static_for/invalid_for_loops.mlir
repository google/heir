// RUN: heir-opt --convert-secret-for-to-static-for --split-input-file --verify-diagnostics %s

func.func @for_loop_with_data_dependent_upper_bound(%arg0: !secret.secret<tensor<32xi16>>, %arg1: !secret.secret<index>) -> !secret.secret<i16> {
    %c0 = arith.constant 0 : index
    %c0_i16 = arith.constant 0 : i16
    %c1 = arith.constant 1 : index
    %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<tensor<32xi16>>, !secret.secret<index>) {
    ^bb0(%arg2: tensor<32xi16>, %arg3: index):
    // unexpected-warning@+1 {{Cannot convert secret scf.for to static affine.for since a static upper bound attribute has not been provided:}}
      %1 = scf.for %arg4 = %c0 to %arg3 step %c1 iter_args(%arg5 = %c0_i16) -> (i16) {
        %extracted = tensor.extract %arg2[%arg4] : tensor<32xi16>
        %2 = arith.addi %extracted, %arg5 : i16
        scf.yield %2 : i16
      }
      secret.yield %1 : i16
    } -> !secret.secret<i16>
    return %0 : !secret.secret<i16>
}

// -----

func.func @for_loop_with_data_dependent_lower_bound(%arg0: !secret.secret<tensor<32xi16>>, %arg1: !secret.secret<index>) -> !secret.secret<i16> {
    %c32 = arith.constant 32 : index
    %c0_i16 = arith.constant 0 : i16
    %c1 = arith.constant 1 : index
    %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<tensor<32xi16>>, !secret.secret<index>) {
    ^bb0(%arg2: tensor<32xi16>, %arg3: index):
    // unexpected-warning@+1 {{Cannot convert secret scf.for to static affine.for since a static lower bound attribute has not been provided:}}
      %1 = scf.for %arg4 = %arg3 to %c32 step %c1 iter_args(%arg5 = %c0_i16) -> (i16) {
        %extracted = tensor.extract %arg2[%arg4] : tensor<32xi16>
        %2 = arith.addi %extracted, %arg5 : i16
        scf.yield %2 : i16
      }
      secret.yield %1 : i16
    } -> !secret.secret<i16>
    return %0 : !secret.secret<i16>
}
