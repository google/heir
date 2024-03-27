// RUN: heir-opt --secretize=entry-function=box_blur --wrap-generic --canonicalize --cse \
// RUN:   --heir-simd-vectorizer %s | FileCheck %s

module  {
  // CHECK-LABEL: @box_blur
  // CHECK-SAME: %[[arg0:.*]]: !secret.secret<tensor<4096xi16>>) -> !secret.secret<tensor<4096xi16>> {
  // CHECK-DAG:    %[[c65_i32:.*]] = arith.constant 65 : i32
  // CHECK-DAG:    %[[c3968:.*]] = arith.constant 3968 : index
  // CHECK-DAG:    %[[c127:.*]] = arith.constant 127 : index
  // CHECK-DAG:    %[[c4032:.*]] = arith.constant 4032 : index
  // CHECK-DAG:    %[[c63:.*]] = arith.constant 63 : index
  // CHECK-DAG:    %[[c4095:.*]] = arith.constant 4095 : index
  // CHECK-DAG:    %[[c4031:.*]] = arith.constant 4031 : index
  // CHECK-DAG:    %[[c64:.*]] = arith.constant 64 : index
  // CHECK-NEXT:   %[[v0:.*]] = secret.generic ins(%[[arg0]] : !secret.secret<tensor<4096xi16>>) {
  // CHECK-NEXT:   ^bb0(%[[arg1:.*]]: tensor<4096xi16>):
  // CHECK-NEXT:     %[[v1:.*]] = tensor_ext.rotate %[[arg1]], %[[c4031]] : tensor<4096xi16>, index
  // CHECK-NEXT:     %[[v2:.*]] = tensor_ext.rotate %[[arg1]], %[[c4095]] : tensor<4096xi16>, index
  // CHECK-NEXT:     %[[v3:.*]] = arith.addi %[[v1]], %[[v2]] : tensor<4096xi16>
  // CHECK-NEXT:     %[[v4:.*]] = tensor_ext.rotate %[[v3]], %[[c64]] : tensor<4096xi16>, index
  // CHECK-NEXT:     %[[v5:.*]] = tensor_ext.rotate %[[arg1]], %[[c127]] : tensor<4096xi16>, index
  // CHECK-NEXT:     %[[v6:.*]] = arith.addi %[[v4]], %[[v5]] : tensor<4096xi16>
  // CHECK-NEXT:     %[[v7:.*]] = arith.addi %[[v6]], %[[arg1]] : tensor<4096xi16>
  // CHECK-NEXT:     %[[v8:.*]] = tensor_ext.rotate %[[v7]], %[[c4032]] : tensor<4096xi16>, index
  // CHECK-NEXT:     %[[v9:.*]] = arith.addi %[[v8]], %[[arg1]] : tensor<4096xi16>
  // CHECK-NEXT:     %[[v10:.*]] = tensor_ext.rotate %[[v9]], %[[c63]] : tensor<4096xi16>, index
  // CHECK-NEXT:     %[[v11:.*]] = arith.addi %[[v10]], %[[v5]] : tensor<4096xi16>
  // CHECK-NEXT:     %[[v12:.*]] = arith.addi %[[v11]], %[[arg1]] : tensor<4096xi16>
  // CHECK-NEXT:     %[[v13:.*]] = tensor_ext.rotate %[[v12]], %[[c3968]] : tensor<4096xi16>, index
  // CHECK-NEXT:     %[[v14:.*]] = tensor_ext.rotate %[[arg1]], %[[c4032]] : tensor<4096xi16>, index
  // CHECK-NEXT:     %[[v15:.*]] = arith.addi %[[v13]], %[[v14]] : tensor<4096xi16>
  // CHECK-NEXT:     %[[v16:.*]] = arith.addi %[[v15]], %[[arg1]] : tensor<4096xi16>
  // CHECK-NEXT:     %[[v17:.*]] = tensor_ext.rotate %[[v16]], %[[c65_i32]] : tensor<4096xi16>, i32
  // CHECK-NEXT:     secret.yield %[[v17]]
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
