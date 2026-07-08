// RUN: heir-opt %s --rotom-assign-layout --mlir-print-local-scope | FileCheck %s

// Matmul: candidates come from the reference's binop pipeline -- replicate
// each operand to the aligned shape, apply the sum roll on either side, align
// the pair by a conversion or a roll, derive the result. The winner becomes
// IR: a rotom.convert_layout / rotom.apply_roll chain on each operand whose
// layout differs from its aligned placement, and one rotom.matmul op naming
// the aligned, compute, and result layouts. The ciphertext lowering reads
// them from the op.
//
// Both operands are seeded sources (data packed at encode time), so the
// search assigns each one its expanded placement directly -- replication
// pieces included -- instead of paying ciphertext conversions from the seed.
//
// At n=64 the whole 4x4x4 iteration space fits one ciphertext's slots, so
// the kernel is one multiply and a k rotate-and-reduce. (At n=16 the
// operands would spread across ciphertexts; see the pipeline execution
// tests.)

#layout_row = #rotom.layout<dims = [#rotom.dim<[G:4:1]>, #rotom.dim<[0:4:1]>, #rotom.dim<[1:4:1]>], n = 64>
#seed_row = #rotom.seed<layouts = [#layout_row]>

module {
  // CHECK: func.func @matmul_assign
  // CHECK-NOT: secret.kernel
  // CHECK-SAME: %arg0: tensor<4x4xf32> {rotom.layout = #rotom.layout<n = 64, dims = {{\[\[0:4:1\], \[1:4:1\], \[R:4:1\]\]}}>
  // CHECK-SAME: %arg1: tensor<4x4xf32> {rotom.layout = #rotom.layout<n = 64, dims = {{\[\[R:4:1\], \[0:4:1\], \[1:4:1\]\]}}>
  // CHECK-SAME: -> (tensor<4x4xf32> {rotom.layout = #rotom.layout<n = 64, dims = {{\[\[0:4:1\], \[G:4:1\], \[1:4:1\]\]}}>})
  func.func @matmul_assign(%a: tensor<4x4xf32> {rotom.seed = #seed_row}, %b: tensor<4x4xf32> {rotom.seed = #seed_row}) -> tensor<4x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<4x4xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x4xf32>) -> tensor<4x4xf32>
    // The winning compute placement is [i][k][j] in one ciphertext's slots
    // (operands repacked to it for free, several placements tie on cost);
    // summing k leaves a slot gap between i and j (the true sums sit at the
    // k=0 offsets).
    // CHECK: rotom.matmul
    // CHECK-SAME: compute = #rotom.layout<n = 64, dims = {{\[\[0:4:1\], \[K:4:1\], \[1:4:1\]\]}}>
    // CHECK-SAME: rotom.layout = #rotom.layout<n = 64, dims = {{\[\[0:4:1\], \[G:4:1\], \[1:4:1\]\]}}>
    // CHECK-SAME: to = #rotom.layout<n = 64, dims = {{\[\[0:4:1\], \[G:4:1\], \[1:4:1\]\]}}>
    // CHECK-NOT: linalg.matmul
    %0 = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%fill : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}

// -----

// Matvec-shaped: the free dim j has extent 1, so no insertion is needed and
// the whole product stays in one ciphertext. Repacking costs a small epsilon
// per source, so the winner keeps the matrix at its row-major seed and
// repacks only the vector (replication where i sits, k innermost); the
// summed k is an innermost slot gap (true sums at the k=0 offsets).

#seed_mat = #rotom.seed<layouts = [#rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:4:1]>], n = 16>]>
#seed_col = #rotom.seed<layouts = [#rotom.layout<dims = [#rotom.dim<[G:4:1]>, #rotom.dim<[0:4:1]>], n = 16>]>

module {
  // CHECK: func.func @matvec_assign
  // CHECK-NOT: secret.kernel
  // CHECK-SAME: %arg0: tensor<4x4xf32> {rotom.layout = #rotom.layout<n = 16, dims = {{\[\[0:4:1\], \[1:4:1\]\]}}>
  // CHECK-SAME: %arg1: tensor<4x1xf32> {rotom.layout = #rotom.layout<n = 16, dims = {{\[\[R:4:1\], \[0:4:1\]\]}}>
  // CHECK-SAME: -> (tensor<4x1xf32> {rotom.layout = #rotom.layout<n = 16, dims = {{\[\[0:4:1\], \[G:4:1\]\]}}>})
  func.func @matvec_assign(%a: tensor<4x4xf32> {rotom.seed = #seed_mat}, %b: tensor<4x1xf32> {rotom.seed = #seed_col}) -> tensor<4x1xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<4x1xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x1xf32>) -> tensor<4x1xf32>
    // CHECK: rotom.matmul
    // CHECK-SAME: compute = #rotom.layout<n = 16, dims = {{\[\[0:4:1\], \[K:4:1\]\]}}>
    // CHECK-SAME: rotom.layout = #rotom.layout<n = 16, dims = {{\[\[0:4:1\], \[G:4:1\]\]}}>
    // CHECK-SAME: to = #rotom.layout<n = 16, dims = {{\[\[0:4:1\], \[G:4:1\]\]}}>
    // CHECK-NOT: linalg.matmul
    %0 = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x1xf32>) outs(%fill : tensor<4x1xf32>) -> tensor<4x1xf32>
    return %0 : tensor<4x1xf32>
  }
}
