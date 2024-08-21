// RUN: heir-opt --convert-secret-extract-to-static-extract %s | FileCheck %s

// CHECK-LABEL: @extract_at_secret_index
func.func @extract_at_secret_index(%arg0: !secret.secret<tensor<32xi16>>, %arg1: !secret.secret<index>) -> !secret.secret<i16> {
    %c0_i16 = arith.constant 0 : i16
    %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<tensor<32xi16>>, !secret.secret<index>) {
    ^bb0(%arg2: tensor<32xi16>, %arg3: index):
      // CHECK: %[[INITAL_VAL:.*]] = tensor.extract %[[TENSOR:.*]][%[[I:.*]]]
      // CHECK-NEXT: %[[FOR:.*]] = affine.for %[[J:.*]] = 0 to 32 iter_args(%[[ARG:.*]] = %[[INITAL_VAL]]) -> (i16) {
      // CHECK-NEXT: %[[COND:.*]] = arith.cmpi eq, %[[J]], %[[SECRET_INDEX:.*]] : index
      // CHECK-NEXT: %[[EXTRACTED:.*]] = tensor.extract %[[TENSOR:.*]][%[[J]]]
      // CHECK: %[[IF:.*]] = scf.if %[[COND:.*]] -> (i16) {
      // CHECK-NEXT:   scf.yield %[[EXTRACTED]] : i16
      // CHECK-NEXT: } else {
      // CHECK-NEXT:   scf.yield %[[ARG]] : i16
      %extracted = tensor.extract %arg2[%arg3] : tensor<32xi16>
      secret.yield %extracted : i16
    } -> !secret.secret<i16>
    return %0 : !secret.secret<i16>
}


// CHECK-LABEL: @extract_and_sum
func.func @extract_and_sum(%arg0: !secret.secret<tensor<32xi16>>, %arg1: !secret.secret<index>) -> !secret.secret<i16> {
    %c0_i16 = arith.constant 0 : i16
    %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<tensor<32xi16>>, !secret.secret<index>) {
    ^bb0(%arg2: tensor<32xi16>, %arg3: index):
      // CHECK: %[[FOR:.*]] = affine.for %[[I:.*]] = 0 to 32
      // CHECK-NEXT: %[[INITAL_VAL:.*]] = tensor.extract %[[TENSOR:.*]][%[[I:.*]]]
      // CHECK-NEXT: %[[INNER_FOR:.*]] = affine.for %[[J:.*]] = 0 to 32 iter_args(%[[ARG:.*]] = %[[INITAL_VAL]]) -> (i16) {
      // CHECK-NEXT: %[[COND:.*]] = arith.cmpi eq, %[[J]], %[[SECRET_INDEX:.*]] : index
      // CHECK-NEXT: %[[EXTRACTED:.*]] = tensor.extract %[[TENSOR:.*]][%[[J]]]
      // CHECK: %[[IF:.*]] = scf.if %[[COND:.*]] -> (i16) {
      // CHECK-NEXT:   scf.yield %[[EXTRACTED]] : i16
      // CHECK-NEXT: } else {
      // CHECK-NEXT:   scf.yield %[[ARG]] : i16
      %1 = affine.for %i = 0 to 32 iter_args(%sum = %c0_i16) -> (i16) {
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
