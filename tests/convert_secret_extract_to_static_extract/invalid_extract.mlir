// RUN: heir-opt --convert-secret-extract-to-static-extract --split-input-file --verify-diagnostics %s

// CHECK-LABEL: @extract_without_size_attr
func.func @extract_without_size_attr(%arg0: !secret.secret<tensor<32xi16>>, %arg1: !secret.secret<index>) -> !secret.secret<i16> {
    %c0_i16 = arith.constant 0 : i16
    %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<tensor<32xi16>>, !secret.secret<index>) {
    ^bb0(%arg2: tensor<32xi16>, %arg3: index):
      %1 = affine.for %i = 0 to 32 iter_args(%sum = %c0_i16) -> (i16) {
        // expected-warning@+1 {{Cannot convert secret tensor.extract to static tensor.extract since a size attribute (`size`) has not been provided on the tensor.extract op:}}
        %extracted = tensor.extract %arg2[%arg3] : tensor<32xi16>
        %extracted_0 = tensor.extract %arg2[%i] : tensor<32xi16>
        %2 = arith.addi %extracted, %extracted_0 : i16
        %3 = arith.addi %sum, %2 : i16
        affine.yield %3 : i16
      }
      secret.yield %1 : i16
    } -> !secret.secret<i16>
    return %0 : !secret.secret<i16>
}

// -----
