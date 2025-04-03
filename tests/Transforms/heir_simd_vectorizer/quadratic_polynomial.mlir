// Ported from: https://github.com/MarbleHE/HECO/blob/3e13744233ab0c09030a41ef98b4e061b6fa2eac/evaluation/comparison/heco_input/quadraticpolynomial_64.mlir

// RUN: heir-opt --secretize --wrap-generic --canonicalize --cse \
// RUN:   --heir-simd-vectorizer %s | FileCheck %s

// CHECK: @quadratic_polynomial
// CHECK: secret.generic
// CHECK-NOT: tensor_ext.rotate
func.func @quadratic_polynomial(%a: tensor<64xi16>, %b: tensor<64xi16>, %c: tensor<64xi16>, %x: tensor<64xi16>, %y: tensor<64xi16>) -> tensor<64xi16> {
  %0 = affine.for %i = 0 to 64 iter_args(%iter = %y) -> (tensor<64xi16>) {
    %ai = tensor.extract %a[%i] : tensor<64xi16>
    %bi = tensor.extract %b[%i] : tensor<64xi16>
    %ci = tensor.extract %c[%i] : tensor<64xi16>
    %xi = tensor.extract %x[%i] : tensor<64xi16>
    %yi = tensor.extract %y[%i] : tensor<64xi16>
    %axi = arith.muli %ai, %xi : i16
    %t1 = arith.addi %axi, %bi : i16
    %t2 = arith.muli %xi, %t1 : i16
    %t3 = arith.addi %t2, %ci : i16
    %t4 = arith.subi %yi, %t3 : i16
    %out = tensor.insert %t4 into %iter[%i] : tensor<64xi16>
    affine.yield %out : tensor<64xi16>
  }
  return %0 : tensor<64xi16>
}
