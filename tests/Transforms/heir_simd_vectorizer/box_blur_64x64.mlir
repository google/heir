// RUN: heir-opt --secretize --wrap-generic --canonicalize --cse \
// RUN:   --heir-simd-vectorizer %s | FileCheck %s

module  {
  // CHECK-LABEL: @box_blur
  // CHECK-SAME: %[[arg0:.*]]: !secret.secret<tensor<4096xi16>>) -> !secret.secret<tensor<4096xi16>> {
  // CHECK-DAG:    %[[c127:.*]] = arith.constant 127 : index
  // CHECK-DAG:    %[[c3968:.*]] = arith.constant 3968 : index
  // CHECK-DAG:    %[[c4032:.*]] = arith.constant 4032 : index
  // CHECK-DAG:    %[[c63:.*]] = arith.constant 63 : index
  // CHECK-DAG:    %[[c65:.*]] = arith.constant 65 : index
  // CHECK-NEXT:   %[[v0:.*]] = secret.generic ins(%[[arg0]] : !secret.secret<tensor<4096xi16>>) {
  // CHECK-NEXT:   ^bb0(%[[arg1:.*]]: tensor<4096xi16>):
  // CHECK-NEXT:     %[[v1:.*]] = tensor_ext.rotate %[[arg1]], %[[c3968]]
  // CHECK-NEXT:     %[[v2:.*]] = tensor_ext.rotate %[[arg1]], %[[c4032]]
  // CHECK-NEXT:     %[[v3:.*]] = arith.addi %[[v1]], %[[v2]]
  // CHECK-NEXT:     %[[v4:.*]] = arith.addi %[[v3]], %[[arg1]]
  // CHECK-NEXT:     %[[v5:.*]] = tensor_ext.rotate %[[v4]], %[[c63]]
  // CHECK-NEXT:     %[[v6:.*]] = arith.addi %[[v5]], %[[v2]]
  // CHECK-NEXT:     %[[v7:.*]] = arith.addi %[[v6]], %[[arg1]]
  // CHECK-NEXT:     %[[v8:.*]] = tensor_ext.rotate %[[v7]], %[[c63]]
  // CHECK-NEXT:     %[[v9:.*]] = tensor_ext.rotate %[[arg1]], %[[c127]]
  // CHECK-NEXT:     %[[v10:.*]] = arith.addi %[[v8]], %[[v9]]
  // CHECK-NEXT:     %[[v11:.*]] = arith.addi %[[v10]], %[[arg1]]
  // CHECK-NEXT:     %[[v12:.*]] = tensor_ext.rotate %[[v11]], %[[c3968]]
  // CHECK-NEXT:     %[[v13:.*]] = arith.addi %[[v12]], %[[v2]]
  // CHECK-NEXT:     %[[v14:.*]] = arith.addi %[[v13]], %[[arg1]]
  // CHECK-NEXT:     %[[v15:.*]] = tensor_ext.rotate %[[v14]], %[[c65]]
  // CHECK-NEXT:     secret.yield %[[v15]]
  // CHECK-NEXT:   } -> !secret.secret<tensor<4096xi16>>
  // CHECK-NEXT:   return %[[v0]]

  func.func @box_blur(%arg0: tensor<4096xi16>) -> tensor<4096xi16> {
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    %0 = affine.for %x = 0 to 64 iter_args(%arg0_x = %arg0) -> (tensor<4096xi16>) {
      %1 = affine.for %y = 0 to 64 iter_args(%arg0_y = %arg0_x) -> (tensor<4096xi16>) {
        %c0_si16 = arith.constant 0 : i16
        %2 = affine.for %j = -1 to 2 iter_args(%value_j = %c0_si16) -> (i16) {
          %6 = affine.for %i = -1 to 2 iter_args(%value_i = %value_j) -> (i16) {
            %7 = arith.addi %x, %i : index
            %8 = arith.muli %7, %c64 : index
            %9 = arith.addi %y, %j : index
            %10 = arith.addi %8, %9 : index
            %11 = arith.remui %10, %c4096 : index
            %12 = tensor.extract %arg0[%11] : tensor<4096xi16>
            %13 = arith.addi %value_i, %12 : i16
            affine.yield %13 : i16
          }
          affine.yield %6 : i16
        }
        %3 = arith.muli %c64, %x : index
        %4 = arith.addi %3, %y : index
        %5 = arith.remui %4, %c4096 : index
        %6 = tensor.insert %2 into %arg0_y[%5] : tensor<4096xi16>
        affine.yield %6 : tensor<4096xi16>
      }
      affine.yield %1 : tensor<4096xi16>
    }
    return %0 : tensor<4096xi16>
  }
}
