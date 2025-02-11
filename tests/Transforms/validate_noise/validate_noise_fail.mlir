// RUN: heir-opt --mlir-to-secret-arithmetic --secret-insert-mgmt-bgv --validate-noise="model=bgv-noise-by-bound-coeff-worst-case-pk plaintext-modulus=4295294977" --verify-diagnostics %s

// This is only for testing whether validate-noise would fail, but
// not for testing the noise model.
// This failing example is handcrafted to fail the noise validation
// using the following conditions:
//   + plaintext modulus is large
//   + modulus is 60bit
//   + consecutive multiplications
//   + using worst-case noise model
//   + When plaintext modulus is large, modulus of 60 bit is not enough
// Note that if any condition changed the test may fail and changes
// to this file are expected

// expected-error@below {{'builtin.module' op Noise validation failed.}}
module {
  func.func @dot_product(%arg0: i16 {secret.secret}) -> i16 {
    %0 = arith.muli %arg0, %arg0 : i16
    %1 = arith.muli %0, %0 : i16
    %2 = arith.muli %1, %1 : i16
    %3 = arith.muli %2, %2 : i16
    %4 = arith.muli %3, %3 : i16
    return %4 : i16
  }
}
