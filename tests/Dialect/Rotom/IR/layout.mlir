// RUN: heir-opt --verify-diagnostics --split-input-file %s

#d0 = #rotom.dim<[0:4:1]>
#d1 = #rotom.dim<[1:4:1]>

// A simple 4x4 layout with 16 slots should have pow2 slot dims.
#layout_ok = #rotom.layout<n = 16, dims = [#d0, #d1]>
func.func private @ok(tensor<16xi32> {foo.bar = #layout_ok})

// -----

// A non-pow2 slot dim size is rejected.
#bad = #rotom.layout<n = 12, dims = [[G:4:1], [0:3:1]]> // expected-error {{slot dim size must be a power of two, got 3}}
func.func private @bad(tensor<16xi32> {foo.bar = #bad})

// -----

// An oversized dim indexes ciphertexts (one element per ciphertext); the
// unused slots are an explicit gap, so the slot side is all gap here.
#split_ok = #rotom.layout<n = 8, dims = [[0:16:1] | [G:8:1]]>
func.func private @split_ok(tensor<16xi32> {foo.bar = #split_ok})

// -----

// The slot side must fill the ciphertext exactly; a written layout may not
// leave capacity implicit.
#underfilled = #rotom.layout<n = 8, dims = [[0:4:1]]> // expected-error {{slot dims must fill the ciphertext exactly (slot extent 4 vs n = 8); spell unused capacity as an explicit gap piece}}
func.func private @underfilled(tensor<16xi32> {foo.bar = #underfilled})

// -----

// The written `|` must sit exactly at the derived boundary: both pieces here
// fit the 8 slots, so neither may be claimed as a ciphertext dim.
#bad_split = #rotom.layout<n = 8, dims = [[0:2:1] | [1:4:1]]> // expected-error {{the written `|` ciphertext/slot split (1 ciphertext dims) does not match the derived split (0): the slot side is the longest dims suffix whose extents fit n = 8}}
func.func private @bad_split(tensor<16xi32> {foo.bar = #bad_split})

// -----

#d0 = #rotom.dim<[0:4:1]>
#d1 = #rotom.dim<[1:4:1]>
// n == 0 is invalid.
#n0_ok = #rotom.layout<n = 0, dims = [#d0, #d1]> // expected-error {{`n` must be > 0, got 0}}
func.func private @n0_ok(tensor<16xi32> {foo.bar = #n0_ok})

// -----

#d0 = #rotom.dim<[0:4:1]>
#d1 = #rotom.dim<[1:2:1]>

// Mismatched extents for a roll pair.
#bad_roll = #rotom.layout<n = 8, rolls = [(0, 1)], dims = [#d0, #d1]> // expected-error {{rolled dims must have the same extent (size)}}
func.func private @bad_roll(tensor<16xi32> {foo.bar = #bad_roll})
