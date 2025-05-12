#ifndef LIB_DIALECT_SECRET_CONVERSIONS_SECRETTOCGGI_SECRETTOCGGI_TD_
#define LIB_DIALECT_SECRET_CONVERSIONS_SECRETTOCGGI_SECRETTOCGGI_TD_

include "mlir/Pass/PassBase.td"

def SecretToCGGI : Pass<"secret-to-cggi"> {
  let summary = "Lower `secret` to `cggi` dialect.";

  let description = [{
    This pass lowers the `secret` dialect to `cggi` dialect.
  }];

  let dependentDialects = [
    "mlir::heir::comb::CombDialect",
    "mlir::heir::cggi::CGGIDialect",
    "mlir::tensor::TensorDialect",
    "mlir::affine::AffineDialect",
  ];
}

#endif  // LIB_DIALECT_SECRET_CONVERSIONS_SECRETTOCGGI_SECRETTOCGGI_TD_
