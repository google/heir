// RUN: heir-opt --split-input-file --verify-diagnostics --polynomial-approximation %s
// RUN: heir-opt --split-input-file --verify-diagnostics --polynomial-approximation=math-exp-method=taylor %s

// A degenerate domain makes the Chebyshev fit meaningless and is the
// precondition rescaleToUnitInterval asserts on, so reject it up front rather
// than dividing by zero when the operand is rescaled onto [-1, 1].
func.func @degenerate_domain(%x: f32 {secret.secret}) -> f32 {
  // expected-error@+1 {{domain_lower must be strictly less than domain_upper}}
  %0 = math.tanh %x {domain_lower = 2.0 : f64, domain_upper = 2.0 : f64} : f32
  return %0 : f32
}

// -----

// An inverted domain would silently mirror the rescale (negative scale factor).
func.func @inverted_domain(%x: f32 {secret.secret}) -> f32 {
  // expected-error@+1 {{domain_lower must be strictly less than domain_upper}}
  %0 = math.tanh %x {domain_lower = 5.0 : f64, domain_upper = -5.0 : f64} : f32
  return %0 : f32
}

// -----

// Same validation on the binary-with-constant path (ConvertBinaryConstOp).
func.func @degenerate_domain_binary(%x: tensor<10xf32> {secret.secret}) -> tensor<10xf32> {
  %c0 = arith.constant dense<0.0> : tensor<10xf32>
  // expected-error@+1 {{domain_lower must be strictly less than domain_upper}}
  %0 = arith.maximumf %x, %c0 {domain_lower = 1.0 : f64, domain_upper = 1.0 : f64} : tensor<10xf32>
  return %0 : tensor<10xf32>
}

// -----

// Inverted domain on math.exp (tested under both Chebyshev and Taylor methods).
func.func @inverted_domain_exp(%x: f32 {secret.secret}) -> f32 {
  // expected-error@+1 {{domain_lower must be strictly less than domain_upper}}
  %0 = math.exp %x {domain_lower = 0.5 : f64, domain_upper = -0.5 : f64} : f32
  return %0 : f32
}

// -----

// Degenerate domain on math.exp.
func.func @degenerate_domain_exp(%x: f32 {secret.secret}) -> f32 {
  // expected-error@+1 {{domain_lower must be strictly less than domain_upper}}
  %0 = math.exp %x {domain_lower = 0.0 : f64, domain_upper = 0.0 : f64} : f32
  return %0 : f32
}

// -----

// Inverted interval via domain_lower with default upper (1.0).
func.func @inverted_domain_default_upper_exp(%x: f32 {secret.secret}) -> f32 {
  // expected-error@+1 {{domain_lower must be strictly less than domain_upper}}
  %0 = math.exp %x {domain_lower = 2.0 : f64} : f32
  return %0 : f32
}

// -----

// Inverted interval via domain_upper with default lower (-1.0).
func.func @inverted_domain_default_lower_exp(%x: f32 {secret.secret}) -> f32 {
  // expected-error@+1 {{domain_lower must be strictly less than domain_upper}}
  %0 = math.exp %x {domain_upper = -2.0 : f64} : f32
  return %0 : f32
}

// -----

// A valid domain must NOT be rejected. --verify-diagnostics fails the test on
// any unexpected diagnostic, so the absence of an expected-error here is the
// assertion that the check does not over-trigger.
func.func @valid_domain(%x: f32 {secret.secret}) -> f32 {
  %0 = math.tanh %x {domain_lower = -3.0 : f64, domain_upper = 3.0 : f64} : f32
  return %0 : f32
}
