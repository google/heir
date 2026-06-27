// RUN: heir-opt %s -split-input-file -verify-diagnostics

// `diagonals` packs the matrix one row per diagonal, so it must be 2D --
// getRotationIndices() and the emitter index its second dim.
func.func @lt_diagonals_not_2d(
    %ctx: !cheddar.context, %ct: !cheddar.ciphertext,
    %evk: !cheddar.evk_map, %diags: tensor<4xf64>) -> !cheddar.ciphertext {
  // expected-error @+1 {{expected `diagonals` to be 2D}}
  %r = cheddar.linear_transform %ctx, %ct, %evk, %diags {diagonal_indices = array<i32: 0>, level = 5 : i64, bs = 1 : i64, gs = 1 : i64} : (!cheddar.context, !cheddar.ciphertext, !cheddar.evk_map, tensor<4xf64>) -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}

// -----

// One diagonal_indices entry per diagonals row.
func.func @lt_row_index_mismatch(
    %ctx: !cheddar.context, %ct: !cheddar.ciphertext,
    %evk: !cheddar.evk_map, %diags: tensor<2x4xf64>) -> !cheddar.ciphertext {
  // expected-error @+1 {{expected one `diagonal_indices` entry per `diagonals` row}}
  %r = cheddar.linear_transform %ctx, %ct, %evk, %diags {diagonal_indices = array<i32: 0>, level = 5 : i64, bs = 2 : i64, gs = 1 : i64} : (!cheddar.context, !cheddar.ciphertext, !cheddar.evk_map, tensor<2x4xf64>) -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}

// -----

// bs/gs must be positive.
func.func @lt_nonpositive_grid(
    %ctx: !cheddar.context, %ct: !cheddar.ciphertext,
    %evk: !cheddar.evk_map, %diags: tensor<2x4xf64>) -> !cheddar.ciphertext {
  // expected-error @+1 {{expected `bs` and `gs` to be >= 1}}
  %r = cheddar.linear_transform %ctx, %ct, %evk, %diags {diagonal_indices = array<i32: 0, 1>, level = 5 : i64, bs = 0 : i64, gs = 1 : i64} : (!cheddar.context, !cheddar.ciphertext, !cheddar.evk_map, tensor<2x4xf64>) -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}

// -----

// The BSGS grid bs*gs must reach the largest diagonal index.
func.func @lt_grid_too_small(
    %ctx: !cheddar.context, %ct: !cheddar.ciphertext,
    %evk: !cheddar.evk_map, %diags: tensor<2x4xf64>) -> !cheddar.ciphertext {
  // expected-error @+1 {{must exceed the largest diagonal index}}
  %r = cheddar.linear_transform %ctx, %ct, %evk, %diags {diagonal_indices = array<i32: 0, 5>, level = 5 : i64, bs = 1 : i64, gs = 1 : i64} : (!cheddar.context, !cheddar.ciphertext, !cheddar.evk_map, tensor<2x4xf64>) -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}

// -----

// eval_poly needs at least one coefficient.
func.func @eval_poly_empty_coeffs(
    %ctx: !cheddar.context, %ct: !cheddar.ciphertext,
    %evk: !cheddar.evk_map) -> !cheddar.ciphertext {
  // expected-error @+1 {{expected a non-empty `coefficients` array}}
  %r = cheddar.eval_poly %ctx, %ct, %evk {coefficients = [], level = 5 : i64, outputLevel = 4 : i64} : (!cheddar.context, !cheddar.ciphertext, !cheddar.evk_map) -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}

// -----

// eval_poly consumes multiplicative depth -- outputLevel cannot exceed level.
func.func @eval_poly_raises_level(
    %ctx: !cheddar.context, %ct: !cheddar.ciphertext,
    %evk: !cheddar.evk_map) -> !cheddar.ciphertext {
  // expected-error @+1 {{cannot raise the level}}
  %r = cheddar.eval_poly %ctx, %ct, %evk {coefficients = [1.0 : f64], level = 4 : i64, outputLevel = 5 : i64} : (!cheddar.context, !cheddar.ciphertext, !cheddar.evk_map) -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}
