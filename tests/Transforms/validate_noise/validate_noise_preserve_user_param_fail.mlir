// RUN: heir-opt --mlir-to-secret-arithmetic --secret-insert-mgmt-bgv --validate-noise=model=bgv-noise-by-bound-coeff-average-case-pk %s --verify-diagnostics

// expected-error@below {{'builtin.module' op Noise validation failed.}}
module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 11, Q = [17], P = [1093633], plaintextModulus = 65537>} {
  func.func @return(%arg0: i16 {secret.secret}) -> i16 {
    %1 = arith.muli %arg0, %arg0 : i16
    return %1 : i16
  }
}
