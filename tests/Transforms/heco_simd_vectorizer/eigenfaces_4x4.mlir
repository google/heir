// Ported from https://github.com/MarbleHE/HECO/blob/3e13744233ab0c09030a41ef98b4e061b6fa2eac/test/filecheck/eigenfaces_small/eigenfaces_1_ssa.mlir

// RUN: heir-opt --secretize --wrap-generic --canonicalize --cse --heco-simd-vectorizer %s | FileCheck %s

// Vectorized over both loops in the 16 slots of %d: %x is spread to the
// slots of each row (three rotations), the sum over each row is a
// rotate-and-reduce (two rotations), and the row sums are moved to the four
// output slots (three rotations). The original HECO needs 14 rotations and
// four extract/insert pairs.
// CHECK: @eigenfaces
// CHECK-NOT: affine.for
// CHECK-COUNT-8: tensor_ext.rotate
// CHECK-NOT: tensor_ext.rotate
// CHECK: tensor.extract_slice %{{.*}}[0] [4] [1] : tensor<16xf64> to tensor<4xf64>
// CHECK-NEXT: secret.yield

func.func @eigenfaces(%x: tensor<4xf64>, %d: tensor<4x4xf64>) -> tensor<4xf64> {
  %c0_sf64 = arith.constant 0.000000e+00 : f64
  %0 = tensor.empty() : tensor<4xf64>
  %1 = affine.for %i = 0 to 4 iter_args(%cur0 = %0) -> (tensor<4xf64>) {
    // squared euclidean distance between x and the i-th row of d
    %2 = affine.for %j = 0 to 4 iter_args(%cur1 = %c0_sf64) -> f64 {
      %3 = tensor.extract %d[%i, %j] : tensor<4x4xf64>
      %4 = arith.mulf %3, %3 : f64
      %5 = tensor.extract %x[%j] : tensor<4xf64>
      %6 = arith.mulf %5, %5 : f64
      %7 = arith.subf %4, %6 : f64
      %8 = arith.addf %cur1, %7 : f64
      affine.yield %8 : f64
    }
    %9 = tensor.insert %2 into %cur0[%i] : tensor<4xf64>
    affine.yield %9 : tensor<4xf64>
  }
  return %1 : tensor<4xf64>
}
