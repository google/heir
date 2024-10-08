// RUN: heir-opt --verify-diagnostics --split-input-file %s | FileCheck %s

// CHECK-NOT: @test_bad_arith_syntax
func.func @test_bad_arith_syntax() {
  %c_vec = arith.constant dense<[1, 2, 1, 2]> : tensor<4xi4>

  // expected-error@+1 {{input bitwidth is required to be in the range [w, 2w], where w is the smallest bit-width that contains the range [0, modulus).}}
  %barrett = mod_arith.barrett_reduce %c_vec { modulus = 17 } : tensor<4xi4>

  return
}

// -----

// CHECK-NOT: @test_bad_mod
func.func @test_bad_mod(%lhs : i8, %rhs : i8) -> i8 {
  // expected-error@+1 {{underlying type's bitwidth must be at least as large as the modulus bitwidth, but got 8 while modulus requires width 23.}}
  %res = mod_arith.add %lhs, %rhs {modulus = 6666666 }: i8
  return %res : i8
}

// -----

// CHECK: @test_bad_mod_warning
func.func @test_bad_mod_warning(%lhs : i8, %rhs : i8) -> i8 {
  // expected-warning@+1 {{for signed (or signless) underlying types, the bitwidth of the underlying type must be at least as large as modulus bitwidth + 1 (for the sign bit), but found 8 while modulus requires width 8.}}
  %res = mod_arith.add %lhs, %rhs {modulus = 135 }: i8
  return %res : i8
}
