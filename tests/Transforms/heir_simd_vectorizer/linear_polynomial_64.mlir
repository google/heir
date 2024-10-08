// Ported from: https://github.com/MarbleHE/HECO/blob/3e13744233ab0c09030a41ef98b4e061b6fa2eac/evaluation/comparison/heco_input/linearpolynomial_64.mlir

// RUN: heir-opt --secretize=entry-function=linear_polynomial --wrap-generic --canonicalize --cse \
// RUN:   --heir-simd-vectorizer %s | FileCheck %s

// CHECK-LABEL: @linear_polynomial
// CHECK: secret.generic
// CHECK-NOT: tensor_ext.rotate
// CHECK-NOT: insert
// CHECK-NOT: extract
func.func @linear_polynomial(%a: tensor<64xi16>, %b: tensor<64xi16>, %x: tensor<64xi16>, %y: tensor<64xi16>) -> tensor<64xi16> {
  %0 = affine.for %i = 0 to 64 iter_args(%iter = %y) -> (tensor<64xi16>) {
    %ai = tensor.extract %a[%i] : tensor<64xi16>
    %bi = tensor.extract %b[%i] : tensor<64xi16>
    %xi = tensor.extract %x[%i] : tensor<64xi16>
    %yi = tensor.extract %y[%i] : tensor<64xi16>
    %axi = arith.muli %ai, %xi : i16
    %t1 = arith.subi %yi, %axi : i16
    %t2 = arith.subi %t1, %bi : i16
    %out = tensor.insert %t2 into %iter[%i] : tensor<64xi16>
    affine.yield %out : tensor<64xi16>
  }
  return %0 : tensor<64xi16>
}
