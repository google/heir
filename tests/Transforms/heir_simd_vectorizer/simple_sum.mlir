// RUN: heir-opt --secretize \
// RUN:   --wrap-generic --canonicalize --cse \
// RUN:   --heir-simd-vectorizer %s | FileCheck %s

// RUN: heir-opt --arith-to-mod-arith --secretize \
// RUN:   --wrap-generic --canonicalize --cse \
// RUN:   --heir-simd-vectorizer %s | FileCheck %s

// Sum all entries of a tensor into a single scalar
// CHECK-LABEL: @simple_sum
// CHECK: secret.generic
// CHECK-COUNT-5: tensor_ext.rotate
// CHECK-NOT: tensor_ext.rotate
func.func @simple_sum(%arg0: tensor<32xi16>) -> i16 {
  %c0 = arith.constant 0 : index
  %c0_si16 = arith.constant 0 : i16
  %0 = affine.for %i = 0 to 32 iter_args(%sum_iter = %c0_si16) -> i16 {
    %1 = tensor.extract %arg0[%i] : tensor<32xi16>
    %2 = arith.addi %1, %sum_iter : i16
    affine.yield %2 : i16
  }
  return %0 : i16
}

// Sum all entries of 4 tensors into a single scalar
// CHECK-LABEL: @simple_sum_nested
// CHECK: secret.generic
// CHECK-COUNT-20: tensor_ext.rotate
// CHECK-NOT: tensor_ext.rotate
// CHECK: tensor.extract
// CHECK-NOT: tensor.extract
func.func @simple_sum_nested(%arg0: tensor<32xi16>, %arg1: tensor<32xi16>, %arg2: tensor<32xi16>, %arg3: tensor<32xi16>) -> i16 {
  %c0_i16 = arith.constant 0 : i16
  %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [1, 32] : tensor<32xi16> into tensor<1x32xi16>
  %expanded_0 = tensor.expand_shape %arg1 [[0, 1]] output_shape [1, 32] : tensor<32xi16> into tensor<1x32xi16>
  %expanded_1 = tensor.expand_shape %arg2 [[0, 1]] output_shape [1, 32] : tensor<32xi16> into tensor<1x32xi16>
  %expanded_2 = tensor.expand_shape %arg3 [[0, 1]] output_shape [1, 32] : tensor<32xi16> into tensor<1x32xi16>
  %concat = tensor.concat dim(0) %expanded, %expanded_0, %expanded_1, %expanded_2 : (tensor<1x32xi16>, tensor<1x32xi16>, tensor<1x32xi16>, tensor<1x32xi16>) -> tensor<4x32xi16>
  %0 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %c0_i16) -> (i16) {
    %extracted_slice = tensor.extract_slice %concat[%arg4, 0] [1, 32] [1, 1] : tensor<4x32xi16> to tensor<32xi16>
    %1 = affine.for %arg6 = 0 to 32 iter_args(%arg7 = %c0_i16) -> (i16) {
      %extracted = tensor.extract %extracted_slice[%arg6] : tensor<32xi16>
      %3 = arith.addi %extracted, %arg7 : i16
      affine.yield %3 : i16
    }
    %2 = arith.addi %1, %arg5 : i16
    affine.yield %2 : i16
  }
  return %0 : i16
}
