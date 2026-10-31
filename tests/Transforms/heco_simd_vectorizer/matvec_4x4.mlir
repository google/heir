// Ported from https://github.com/MarbleHE/HECO/blob/3e13744233ab0c09030a41ef98b4e061b6fa2eac/test/filecheck/matvec/matvec_1_ssa.mlir

// RUN: heir-opt --wrap-generic --canonicalize --cse --heco-simd-vectorizer %s | FileCheck %s

// The diagonal method: two rotations of %v (the original HECO needs twelve),
// and two rotations of plaintext diagonals before the secret.generic.
// CHECK: @matvec
// CHECK: affine.for
// CHECK:   affine.for
// CHECK:     tensor.insert %{{.*}} : tensor<4x4xf64>
// CHECK-COUNT-2: tensor_ext.rotate
// CHECK: secret.generic
// CHECK: ^body(%[[v:.*]]: tensor<4xf64>):
// CHECK: tensor_ext.rotate %[[v]], %c1
// CHECK: tensor_ext.rotate
// CHECK-NOT: tensor_ext.rotate
// CHECK: secret.yield

func.func @matvec(%m: tensor<16xf64>, %v: tensor<4xf64> {secret.secret}) -> tensor<4xf64> {
  %c4 = arith.constant 4 : index
  %c0_sf64 = arith.constant 0.000000e+00 : f64
  // for each row in matrix
  %0 = affine.for %i = 0 to 4 iter_args(%r = %v) -> (tensor<4xf64>) {
    // iterate over the vector
    %1 = affine.for %j = 0 to 4 iter_args(%sum = %c0_sf64) -> (f64) {
      %2 = arith.muli %i, %c4 : index
      %ij = arith.addi %2, %j : index
      %mij = tensor.extract %m[%ij] : tensor<16xf64>
      %vj = tensor.extract %v[%j] : tensor<4xf64>
      %p = arith.mulf %mij, %vj : f64
      %s = arith.addf %sum, %p : f64
      affine.yield %s : f64
    }
    %vnew = tensor.insert %1 into %r[%i] : tensor<4xf64>
    affine.yield %vnew : tensor<4xf64>
  }
  return %0 : tensor<4xf64>
}
