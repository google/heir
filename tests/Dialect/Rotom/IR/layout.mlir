// RUN: heir-opt --verify-diagnostics --split-input-file %s

#d0 = #rotom.dim<dim = 0, size = 4, stride = 1>
#d1 = #rotom.dim<dim = 1, size = 4, stride = 1>

// A simple 4x4 layout with 16 slots should have pow2 slot dims.
#layout_ok = #rotom.layout<dims = [#d0, #d1], n = 16>
func.func private @ok(tensor<16xi32> {foo.bar = #layout_ok})

// -----

// This forces a non-pow2 slot dim size after splitting: size=3 divides 12 but not pow2.
#bad = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 3, stride = 1>], n = 12> // expected-error {{slot dim size must be a power of two, got 3}}
func.func private @bad(tensor<16xi32> {foo.bar = #bad})

// -----

// Splitting case: size > n causes a ct/slot split, and slot-side size becomes n (must be pow2).
#split_ok = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 16, stride = 1>], n = 8>
func.func private @split_ok(tensor<16xi32> {foo.bar = #split_ok})

// -----

// size > n but not divisible => verifier error (mirrors Python assert size % n == 0).
#split_bad = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 10, stride = 1>], n = 8> // expected-error {{dim size 10 must be divisible by remaining slot capacity 8}}
func.func private @split_bad(tensor<16xi32> {foo.bar = #split_bad})

// -----

#d0 = #rotom.dim<dim = 0, size = 4, stride = 1>
#d1 = #rotom.dim<dim = 1, size = 4, stride = 1>
// n == 0 is invalid.
#n0_ok = #rotom.layout<dims = [#d0, #d1], n = 0> // expected-error {{`n` must be > 0, got 0}}
func.func private @n0_ok(tensor<16xi32> {foo.bar = #n0_ok})

// -----

#d0 = #rotom.dim<dim = 0, size = 4, stride = 1>
#d1 = #rotom.dim<dim = 1, size = 2, stride = 1>

// Mismatched extents for a roll pair.
#bad_roll = #rotom.layout<dims = [#d0, #d1], n = 8, rolls = [0, 1]> // expected-error {{rolled dims must have the same extent (size)}}
func.func private @bad_roll(tensor<16xi32> {foo.bar = #bad_roll})
