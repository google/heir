// RUN: heir-opt --canonicalize %s | FileCheck %s

!Z17_i64 = !mod_arith.int<17 : i64>
// CHECK: test_dense_folding
// TODO(#1759): support actually folding this
func.func @test_dense_folding() -> tensor<2x!Z17_i64> {
  %0 = mod_arith.constant dense<[2, 2]> : tensor<2x!Z17_i64>
  %1 = mod_arith.constant dense<[0, 0]> : tensor<2x!Z17_i64>
  %2 = mod_arith.add %0, %1 : tensor<2x!Z17_i64>
  return %2 : tensor<2x!Z17_i64>
}
