// The mlir-to-rotom-ciphertext pipeline takes a high-level two-layer MLP --
// h = W1 x + b1; a = h .* h (square activation); y = W2 a + b2 -- with NO
// hand-written layouts and auto-discovers the ciphertext-axis diagonal/BSGS
// matvec kernel for both layers, lowering the whole network to ciphertext
// rotations with no leftover linalg.matmul. This is the end-to-end front-end:
// seed -> assign -> materialize -> convert, in one command.

// RUN: heir-opt %s --mlir-to-rotom-ciphertext=ciphertext-size=4 | FileCheck %s
// RUN: heir-opt %s --mlir-to-rotom-ciphertext=ciphertext-size=4 | FileCheck %s --check-prefix=ROTATE

// Both matmuls must lower (the diagonal kernel fired for each layer); a leftover
// linalg.matmul would mean a layer was not auto-discovered.
// CHECK-LABEL: func.func @main
// CHECK-NOT: linalg.matmul

// The diagonal/BSGS kernel realizes the contraction as ciphertext rotations.
// ROTATE: tensor_ext.rotate

module {
  func.func @main(%x: !secret.secret<tensor<4x1xf32>>,
                  %w1: tensor<4x4xf32>, %b1: tensor<4x1xf32>,
                  %w2: tensor<2x4xf32>, %b2: tensor<2x1xf32>)
                  -> !secret.secret<tensor<2x1xf32>> {
    %0 = secret.generic(%x : !secret.secret<tensor<4x1xf32>>) {
    ^bb0(%xp: tensor<4x1xf32>):
      %h = linalg.matmul ins(%w1, %xp : tensor<4x4xf32>, tensor<4x1xf32>) outs(%b1 : tensor<4x1xf32>) -> tensor<4x1xf32>
      %a = arith.mulf %h, %h : tensor<4x1xf32>
      %y = linalg.matmul ins(%w2, %a : tensor<2x4xf32>, tensor<4x1xf32>) outs(%b2 : tensor<2x1xf32>) -> tensor<2x1xf32>
      secret.yield %y : tensor<2x1xf32>
    } -> !secret.secret<tensor<2x1xf32>>
    return %0 : !secret.secret<tensor<2x1xf32>>
  }
}
