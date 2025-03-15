// RUN: heir-opt --mlir-to-secret-arithmetic --secret-insert-mgmt-bgv --validate-noise=model=bgv-noise-kpz21 %s --verify-diagnostics --split-input-file

// expected-error@below {{'builtin.module' op The level in the scheme param is smaller than the max level.}}
module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 11, Q = [17], P = [1093633], plaintextModulus = 65537>} {
  func.func @return(%arg0: i16 {secret.secret}) -> i16 {
    %1 = arith.muli %arg0, %arg0 : i16
    return %1 : i16
  }
}

// -----

// expected-error@below {{'builtin.module' op Noise validation failed.}}
module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 11, Q = [17, 23], P = [1093633], plaintextModulus = 65537>} {
  func.func @return(%arg0: i16 {secret.secret}) -> i16 {
    %1 = arith.muli %arg0, %arg0 : i16
    return %1 : i16
  }
}
