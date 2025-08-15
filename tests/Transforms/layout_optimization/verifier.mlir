// RUN: heir-opt --split-input-file --layout-optimization --verify-diagnostics %s

// Valid
func.func @main(%arg0: tensor<512xf32>) -> tensor<512xf32> {
  %0 = arith.addf %arg0, %arg0 {secret.kernel = #secret.kernel<name="Trivial", force=false>} : tensor<512xf32>
  func.return %0 : tensor<512xf32>
}

// -----

// Bad kernel name
func.func @main(%arg0: tensor<512xf32>) -> tensor<512xf32> {
  // expected-error@below {{has unsupported kernel}}
  %0 = arith.addf %arg0, %arg0 {secret.kernel = #secret.kernel<name="MatvecNaive", force=false>} : tensor<512xf32>
  func.return %0 : tensor<512xf32>
}

// -----

// Good kernel name
func.func @main(%arg0: tensor<512x512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
  %cst = tensor.empty() : tensor<512xf32>
  %0 = linalg.matvec
    {secret.kernel = #secret.kernel<name="MatvecDiagonal", force=false>}
    ins(%arg0, %arg1 : tensor<512x512xf32>, tensor<512xf32>)
    outs(%cst : tensor<512xf32>) -> tensor<512xf32>
  func.return %0 : tensor<512xf32>
}
// -----

// Missing required kernel
func.func @main(%arg0: tensor<512x512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
  %cst = tensor.empty() : tensor<512xf32>
  // expected-error@below {{has unsupported kernel}}
  %0 = linalg.matvec
    ins(%arg0, %arg1 : tensor<512x512xf32>, tensor<512xf32>)
    outs(%cst : tensor<512xf32>) -> tensor<512xf32>
  func.return %0 : tensor<512xf32>
}
