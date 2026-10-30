// RUN: heir-opt --secretize --wrap-generic --canonicalize --cse \
// RUN:   --heco-simd-vectorizer %s | FileCheck %s

module  {
  // CHECK: @box_blur
  // CHECK-SAME: %[[arg0:.*]]: !secret.secret<tensor<256xi16>>) -> !secret.secret<tensor<256xi16>> {
  // CHECK-DAG:    %[[c239:.*]] = arith.constant 239 : index
  // CHECK-DAG:    %[[c1:.*]] = arith.constant 1 : index
  // CHECK-DAG:    %[[c16:.*]] = arith.constant 16 : index
  // CHECK-DAG:    %[[c32:.*]] = arith.constant 32 : index
  // CHECK-DAG:    %[[c2:.*]] = arith.constant 2 : index
  // CHECK-NEXT:   %[[v0:.*]] = secret.generic(%[[arg0]]: !secret.secret<tensor<256xi16>>) {
  // CHECK-NEXT:   ^body(%[[arg1:.*]]: tensor<256xi16>):
  // CHECK-NEXT:     %[[v1:.*]] = tensor_ext.rotate %[[arg1]], %[[c16]]
  // CHECK-NEXT:     %[[v2:.*]] = tensor_ext.rotate %[[arg1]], %[[c32]]
  // CHECK-NEXT:     %[[v3:.*]] = arith.addi %[[v1]], %[[v2]]
  // CHECK-NEXT:     %[[v4:.*]] = arith.addi %[[arg1]], %[[v3]]
  // CHECK-NEXT:     %[[v5:.*]] = tensor_ext.rotate %[[v4]], %[[c1]]
  // CHECK-NEXT:     %[[v6:.*]] = tensor_ext.rotate %[[v4]], %[[c2]]
  // CHECK-NEXT:     %[[v7:.*]] = arith.addi %[[v5]], %[[v6]]
  // CHECK-NEXT:     %[[v8:.*]] = arith.addi %[[v4]], %[[v7]]
  // CHECK-NEXT:     %[[v9:.*]] = tensor_ext.rotate %[[v8]], %[[c239]]
  // CHECK-NEXT:     secret.yield %[[v9]]
  // CHECK-NEXT:   } -> !secret.secret<tensor<256xi16>>
  // CHECK-NEXT:   return %[[v0]]

  func.func @box_blur(%arg0: tensor<256xi16>) -> tensor<256xi16> {
    %c256 = arith.constant 256 : index
    %c16 = arith.constant 16 : index
    %0 = affine.for %x = 0 to 16 iter_args(%arg0_x = %arg0) -> (tensor<256xi16>) {
      %1 = affine.for %y = 0 to 16 iter_args(%arg0_y = %arg0_x) -> (tensor<256xi16>) {
        %c0_si16 = arith.constant 0 : i16
        %2 = affine.for %j = -1 to 2 iter_args(%value_j = %c0_si16) -> (i16) {
          %6 = affine.for %i = -1 to 2 iter_args(%value_i = %value_j) -> (i16) {
            %7 = arith.addi %x, %i : index
            %8 = arith.muli %7, %c16 : index
            %9 = arith.addi %y, %j : index
            %10 = arith.addi %8, %9 : index
            %11 = arith.remui %10, %c256 : index
            %12 = tensor.extract %arg0[%11] : tensor<256xi16>
            %13 = arith.addi %value_i, %12 : i16
            affine.yield %13 : i16
          }
          affine.yield %6 : i16
        }
        %3 = arith.muli %c16, %x : index
        %4 = arith.addi %3, %y : index
        %5 = arith.remui %4, %c256 : index
        %6 = tensor.insert %2 into %arg0_y[%5] : tensor<256xi16>
        affine.yield %6 : tensor<256xi16>
      }
      affine.yield %1 : tensor<256xi16>
    }
    return %0 : tensor<256xi16>
  }
}
