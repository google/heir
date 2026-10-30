// RUN: heir-opt --secretize --wrap-generic --canonicalize --cse --heco-simd-vectorizer %s | FileCheck %s

// The 64x64 version of box_blur_16x16, which was too slow to compile while
// the pipeline fully unrolled loops.
// CHECK: @box_blur
// CHECK-NOT: affine.for
// CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[c2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[c64:.*]] = arith.constant 64 : index
// CHECK-DAG: %[[c128:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[c4031:.*]] = arith.constant 4031 : index
// CHECK: tensor_ext.rotate %{{.*}}, %[[c64]]
// CHECK: tensor_ext.rotate %{{.*}}, %[[c128]]
// CHECK: tensor_ext.rotate %{{.*}}, %[[c1]]
// CHECK: tensor_ext.rotate %{{.*}}, %[[c2]]
// CHECK: tensor_ext.rotate %{{.*}}, %[[c4031]]
// CHECK-NOT: tensor_ext.rotate

module  {
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
