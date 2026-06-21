// A neural-net layer y = x . W^T is a vector-times-matrix matmul
// x(1xK) * Wt(KxN) (here Wt = transpose(W)). The ciphertext-axis diagonal matvec
// kernel only handles matrix * column-vector, so rotom-seed-layout rewrites the
// layer to (W . x^T)^T before seeding. After assignment the matmul must carry
// the RotomMatmul kernel (i.e. the diagonal was discovered) -- a leftover
// row-major matmul would mean the orientation was not normalized.

// RUN: heir-opt %s --rotom-seed-layout=n=16 --rotom-assign-layout --mlir-print-local-scope | FileCheck %s
// RUN: heir-opt %s --rotom-seed-layout=n=16 --rotom-assign-layout --mlir-print-local-scope | FileCheck %s --check-prefix=ROLL

// CHECK-LABEL: func.func @main
// The matmul is normalized + lowered with the diagonal kernel.
// CHECK: secret.kernel = #secret.kernel<name = "RotomMatmul"
// The matrix operand is packed as a rolled (diagonal) layout.
// ROLL: rolls = [(0, 1)]

// W is 4x8 (output 4 <= contraction 8), so after the rewrite the matrix W . x^T
// is a squat 4x8 matvec the diagonal kernel handles (a tall result N > K is the
// pre-existing unsupported case).
module {
  func.func @main(%x: !secret.secret<tensor<1x8xf32>>, %w: tensor<4x8xf32>, %b: tensor<1x4xf32>) -> !secret.secret<tensor<1x4xf32>> {
    %0 = secret.generic(%x : !secret.secret<tensor<1x8xf32>>) {
    ^bb0(%xp: tensor<1x8xf32>):
      %empty = tensor.empty() : tensor<8x4xf32>
      %wt = linalg.transpose ins(%w : tensor<4x8xf32>) outs(%empty : tensor<8x4xf32>) permutation = [1, 0]
      %y = linalg.matmul ins(%xp, %wt : tensor<1x8xf32>, tensor<8x4xf32>) outs(%b : tensor<1x4xf32>) -> tensor<1x4xf32>
      secret.yield %y : tensor<1x4xf32>
    } -> !secret.secret<tensor<1x4xf32>>
    return %0 : !secret.secret<tensor<1x4xf32>>
  }
}
