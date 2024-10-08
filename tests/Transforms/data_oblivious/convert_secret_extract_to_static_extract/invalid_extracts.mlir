// RUN: heir-opt --convert-secret-extract-to-static-extract --split-input-file --verify-diagnostics %s

// CHECK-LABEL: @multi_dimensional_extract
func.func @multi_dimensional_extract(%arg0: !secret.secret<tensor<32x32xi16>>, %arg1: !secret.secret<index>) -> !secret.secret<i16> {
    %c0_i16 = arith.constant 0 : i16
    %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<tensor<32x32xi16>>, !secret.secret<index>) {
    ^bb0(%arg2: tensor<32x32xi16>, %arg3: index):
      %1 = affine.for %i = 0 to 32 iter_args(%sum = %c0_i16) -> (i16) {
        // expected-warning@+1 {{Currently, transformation only supports 1D tensors:}}
        %extracted = tensor.extract %arg2[%arg3, %arg3] : tensor<32x32xi16>
        // expected-warning@+1 {{Currently, transformation only supports 1D tensors:}}
        %extracted_0 = tensor.extract %arg2[%i, %i] : tensor<32x32xi16>
        %2 = arith.addi %extracted, %extracted_0 : i16
        %3 = arith.addi %sum, %2 : i16
        affine.yield %3 : i16
      }
      secret.yield %1 : i16
    } -> !secret.secret<i16>
    return %0 : !secret.secret<i16>
}

// -----
