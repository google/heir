// Ported from https://github.com/MarbleHE/HECO/blob/ba027a51f4e0a376a19ca05b1dcc7ab76da78a3e/evaluation/comparison/heco_input/gxkernel_64x64.mlir

// RUN: heir-opt --secretize --wrap-generic --canonicalize --cse \
// RUN:   --heir-simd-vectorizer %s | FileCheck %s

// CHECK: @gx_kernel
// CHECK: secret.generic
// CHECK-COUNT-6: tensor_ext.rotate
// CHECK-NOT: tensor_ext.rotate
func.func @gx_kernel(%arg0: tensor<256xi16>) -> tensor<256xi16> {
  %c256 = arith.constant 256 : index
  %c16 = arith.constant 16 : index
  %c1_index = arith.constant 1 : index
  %c0_si16 = arith.constant 0 : i16
  %c0 = arith.constant 0 : i16
  %c1 = arith.constant 1 : i16
  %c2 = arith.constant 2 : i16
  %cm1= arith.constant -1 : i16
  %cm2 = arith.constant -2 : i16
  %weight_matrix = tensor.from_elements %c1, %cm1, %c2, %cm2, %c1, %cm1, %c0, %c0, %c0 : tensor<3x3xi16>
  %0 = affine.for %x = 0 to 16 iter_args(%arg0_x = %arg0) -> (tensor<256xi16>) {
    %1 = affine.for %y = 0 to 16 iter_args(%arg0_y = %arg0_x) -> (tensor<256xi16>) {
      %2 = affine.for %j = -1 to 2 iter_args(%value_j = %c0_si16) -> (i16) {
        %6 = affine.for %i = -1 to 2 iter_args(%value_i = %value_j) -> (i16) {
          %7 = arith.addi %x, %i : index
          %8 = arith.muli %7, %c16 : index
          %9 = arith.addi %y, %j : index
          %10 = arith.addi %8, %9 : index
          %11 = arith.remui %10, %c256 : index
          %12 = tensor.extract %arg0[%11] : tensor<256xi16>
          // Get the weight from the weight matrix!
          %ip = arith.addi %i,%c1_index : index
          %jp = arith.addi %j,%c1_index : index
          %w = tensor.extract %weight_matrix[%ip,%jp] : tensor<3x3xi16>
          %mul = arith.muli %12, %w : i16
          %13 = arith.addi %value_i, %mul : i16
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
