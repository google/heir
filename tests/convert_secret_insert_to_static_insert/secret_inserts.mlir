// RUN: heir-opt --convert-secret-insert-to-static-insert %s | FileCheck %s

// CHECK-LABEL: @insert_to_secret_index
func.func @insert_to_secret_index(%arg0: !secret.secret<tensor<16xi32>>, %arg1: !secret.secret<index>) -> !secret.secret<tensor<16xi32>> {
  %c10_i32 = arith.constant 10 : i32
  %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<tensor<16xi32>>, !secret.secret<index>) {
  ^bb0(%arg2: tensor<16xi32>, %arg3: index):
    // CHECK: %[[FOR:.*]] = affine.for %[[I:.*]] = 0 to 16
    // CHECK:      %[[INSERTED:.*]] = tensor.insert
    // CHECK-NEXT: %[[IF:.*]] = scf.if %[[COND:.*]] -> (tensor<16xi32>) {
    // CHECK-NEXT:   scf.yield %[[INSERTED]] : tensor<16xi32>
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   scf.yield %[[OLD_TENSOR:.*]] : tensor<16xi32>
    // CHECK-NEXT: }
    %inserted = tensor.insert %c10_i32 into %arg2[%arg3] : tensor<16xi32>
    secret.yield %inserted : tensor<16xi32>
  } -> !secret.secret<tensor<16xi32>>
  return %0 : !secret.secret<tensor<16xi32>>
}

// CHECK-LABEL: @insert_and_sum
func.func @insert_and_sum(%arg0: !secret.secret<tensor<32xi16>>, %arg1: !secret.secret<index>) -> !secret.secret<tensor<32xi16>> {
    %c0_i16 = arith.constant 0 : i16
    %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<tensor<32xi16>>, !secret.secret<index>) {
    ^bb0(%arg2: tensor<32xi16>, %arg3: index):
        // CHECK: %[[FOR:.*]]:2 = affine.for %[[I:.*]] = 0 to 32
        // CHECK-NEXT:  %[[EXTRACTED:.*]] = tensor.extract
        // CHECK-NEXT:  %[[SUM:.*]] = arith.addi
        // CHECK-NEXT:  %[[INNER_FOR:.*]] = affine.for %[[J:.*]] = 0 to 32 iter_args(%[[TENSOR:.*]] = %[[INITAL_TENSOR:.*]]) -> (tensor<32xi16>)
        // CHECK-NEXT:      %[[COND:.*]] = arith.cmpi
        // CHECK-NEXT:      %[[INSERTED:.*]] = tensor.insert %[[SUM]] into %[[TENSOR]][%[[J]]]
        // CHECK-NEXT:      %[[IF:.*]] = scf.if %[[COND]] -> (tensor<32xi16>) {
        // CHECK-NEXT:        scf.yield %[[INSERTED]] : tensor<32xi16>
        // CHECK-NEXT:      } else {
        // CHECK-NEXT:        scf.yield %[[TENSOR]] : tensor<32xi16>
        // CHECK-NEXT:      }
        // CHECK: %[[FINAL_TENSOR:.*]] = affine.for %[[X:.*]] = 0 to 32 iter_args(%[[FOR_TENSOR:.*]] = %[[FOR]]#1) -> (tensor<32xi16>)
        // CHECK-NEXT:      %[[SECOND_COND:.*]] = arith.cmpi
        // CHECK-NEXT:      %[[FINAL_INSERTED:.*]] = tensor.insert %[[FOR]]#0 into %[[FOR_TENSOR]][%[[X]]]
        // CHECK-NEXT:      %[[TENSOR:.*]] = scf.if %[[SECOND_COND]] -> (tensor<32xi16>) {
        // CHECK-NEXT:        scf.yield %[[FINAL_INSERTED]] : tensor<32xi16>
        // CHECK-NEXT:      } else {
        // CHECK-NEXT:        scf.yield %[[FOR_TENSOR]] : tensor<32xi16>
        // CHECK-NEXT:      }
        %1, %newTensor = affine.for %i = 0 to 32 iter_args(%arg = %c0_i16, %tensor = %arg2) -> (i16, tensor<32xi16>) {
        %extracted = tensor.extract %tensor[%i] : tensor<32xi16>
        %sum = arith.addi %arg, %extracted : i16
        %inserted = tensor.insert %sum into %tensor[%arg3] : tensor<32xi16>
        affine.yield %sum, %inserted : i16, tensor<32xi16>
      }

      %finalTensor = tensor.insert %1 into %newTensor[%arg3] : tensor<32xi16>

      secret.yield %finalTensor : tensor<32xi16>
    } -> !secret.secret<tensor<32xi16>>
    return %0 : !secret.secret<tensor<32xi16>>
}
