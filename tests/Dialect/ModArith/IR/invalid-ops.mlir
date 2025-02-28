// RUN: heir-opt --verify-diagnostics --split-input-file %s | FileCheck %s

!Zp = !mod_arith.int<255 : i8>

// -----

!Zp = !mod_arith.int<255 : i32>

// CHECK-NOT: @test_bad_extract
func.func @test_bad_extract(%lhs : !Zp) -> i8 {
  // expected-error@+1 {{the result integer type should be of the same width as the mod arith type width, but got 8 while mod arith type width 32}}
  %m = mod_arith.extract %lhs : !Zp -> i8
  return %m : i8
}

// -----

// CHECK-NOT: @test_bad_arith_syntax
func.func @test_bad_arith_syntax() {
  %c_vec = arith.constant dense<[1, 2, 1, 2]> : tensor<4xi4>

  // expected-error@+1 {{input bitwidth is required to be in the range [w, 2w], where w is the smallest bit-width that contains the range [0, modulus).}}
  %barrett = mod_arith.barrett_reduce %c_vec { modulus = 17 } : tensor<4xi4>

  return
}

// -----

// CHECK-NOT: @test_barrett_neg_mod_err
func.func @test_barrett_neg_mod_err(%arg : i8) -> i8 {
  // expected-error@+1 {{provided modulus -3 is not a positive integer.}}
  %res = mod_arith.barrett_reduce %arg { modulus = -3 : i7 } : i8
  return %res : i8
}

// -----
// CHECK-NOT @test_constant_bad_width
func.func @test_constant_bad_width() {
  // expected-error@+1 {{value's bitwidth must not be larger than underlying type.}}
  %c = mod_arith.constant 512 : !mod_arith.int<17 : i8>
  return
}
