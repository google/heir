// RUN: heir-opt --verify-diagnostics --split-input-file %s | FileCheck %s

// CHECK-NOT: @test_bad_arith_syntax
func.func @test_bad_arith_syntax() {
  %c_vec = arith.constant dense<[1, 2, 1, 2]> : tensor<4xi4>

  // expected-error@+1 {{input bitwidth is required to be in the range [w, 2w], where w is the smallest bit-width that contains the range [0, modulus).}}
  %barrett = arith_ext.barrett_reduce %c_vec { modulus = 17 } : tensor<4xi4>

  return
}

// -----
