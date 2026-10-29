// RUN: heir-opt --split-input-file --verify-diagnostics %s

// Negative case: even modulus
!Zp_even = !mod_arith.int<16 : i32>

func.func @test_even_q(%a: !Zp_even, %b: !Zp_even) {
  // expected-error@below {{modulus must be odd}}
  %0 = mod_arith.mont_mul %a, %b : !Zp_even
  return
}

// -----

// Negative case: invalid storage width W = 16
!Zp_w16 = !mod_arith.int<17 : i16>

func.func @test_w16(%a: !Zp_w16, %b: !Zp_w16) {
  // expected-error@below {{underlying bitwidth must be 32 or 64}}
  %0 = mod_arith.mont_mul %a, %b : !Zp_w16
  return
}

// -----

// Negative case: modulus too large (>= 2^(W-1))
// Type verification rejects 2147483649:i32 because it requires 32 bits, but underlying type width is 32.
// expected-error@below {{underlying type's bitwidth must be 1 bit larger than the modulus bitwidth, but got 32 while modulus requires width 32}}
!Zp_too_large = !mod_arith.int<2147483649 : i32>

func.func @test_q_too_large(%a: !Zp_too_large, %b: !Zp_too_large) {
  %0 = mod_arith.mont_mul %a, %b : !Zp_too_large
  return
}

// -----

// Negative case: mont_reduce wide input bitwidth mismatch (not 2*W)
!Zp_32 = !mod_arith.int<974258177 : i32>

func.func @test_reduce_width_mismatch(%w: i32) {
  // expected-error@below {{input type must have bitwidth 64}}
  %0 = mod_arith.mont_reduce %w : i32 -> !Zp_32
  return
}
