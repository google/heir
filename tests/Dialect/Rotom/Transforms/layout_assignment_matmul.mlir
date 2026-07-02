// RUN: heir-opt %s --rotom-assign-layout --mlir-print-local-scope | FileCheck %s

// Roll-free matmul: candidates come from the deterministic
// align -> multiply -> sum-k plans (ContractionAlignment), so the matmul gets
// a layout but -- unlike the elementwise ops -- never a secret.kernel. The
// chosen (lhs, rhs, result) layouts are recorded under rotom.matmul, from
// which the ciphertext lowering re-derives the plan.
//
// At n=64 the whole 4x4x4 iteration space fits one ciphertext's slots, so
// every plan is a same-shape conversion. (At n=16 the operands spread across
// ciphertexts instead; that expansion is priced and emitted as explicit
// rotate/mask/accumulate steps -- see the pipeline execution test.)

#layout_row = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:4:1]>], n = 64>
#seed_row = #rotom.seed<layouts = [#layout_row]>

module {
  // CHECK: func.func @matmul_assign
  // CHECK-NOT: secret.kernel
  // CHECK-SAME: -> (tensor<4x4xf32> {rotom.layout = #rotom.layout<n = 64, dims = {{\[\[0:4:1\], \[-2:4:1\], \[1:4:1\]\]}}>})
  func.func @matmul_assign(%a: tensor<4x4xf32> {rotom.seed = #seed_row}, %b: tensor<4x4xf32> {rotom.seed = #seed_row}) -> tensor<4x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<4x4xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x4xf32>) -> tensor<4x4xf32>
    // The winning plan hosts the lhs and appends j innermost in the slots;
    // summing k leaves a gap between i and j (the true sums sit at the
    // k=0 offsets).
    // CHECK: linalg.matmul
    // CHECK-SAME: rotom.layout = #rotom.layout<n = 64, dims = {{\[\[0:4:1\], \[-2:4:1\], \[1:4:1\]\]}}>
    // CHECK-SAME: rotom.matmul
    %0 = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%fill : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}

// -----

// Matvec-shaped: the free dim j has extent 1, so no insertion is needed and
// the whole product stays in one ciphertext -- i in the slots, the summed k
// left as an innermost gap (true sums at the k=0 offsets).

#seed_mat = #rotom.seed<layouts = [#rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:4:1]>], n = 16>]>
#seed_col = #rotom.seed<layouts = [#rotom.layout<dims = [#rotom.dim<[0:4:1]>], n = 16>]>

module {
  // CHECK: func.func @matvec_assign
  // CHECK-NOT: secret.kernel
  // CHECK-SAME: -> (tensor<4x1xf32> {rotom.layout = #rotom.layout<n = 16, dims = {{\[\[0:4:1\], \[-2:4:1\]\]}}>})
  func.func @matvec_assign(%a: tensor<4x4xf32> {rotom.seed = #seed_mat}, %b: tensor<4x1xf32> {rotom.seed = #seed_col}) -> tensor<4x1xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<4x1xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x1xf32>) -> tensor<4x1xf32>
    // CHECK: linalg.matmul
    // CHECK-SAME: rotom.layout = #rotom.layout<n = 16, dims = {{\[\[0:4:1\], \[-2:4:1\]\]}}>
    // CHECK-SAME: rotom.matmul
    %0 = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x1xf32>) outs(%fill : tensor<4x1xf32>) -> tensor<4x1xf32>
    return %0 : tensor<4x1xf32>
  }
}
