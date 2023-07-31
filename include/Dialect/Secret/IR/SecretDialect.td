#ifndef HEIR_INCLUDE_DIALECT_SECRET_IR_SECRETDIALECT_TD_
#define HEIR_INCLUDE_DIALECT_SECRET_IR_SECRETDIALECT_TD_

include "mlir/IR/DialectBase.td"

def Secret_Dialect : Dialect {
  let name = "secret";
  let summary = "A dialect for computations that operate on encrypted secrets.";
  let description = [{
    Secret is intended to serve as a scheme-agnostic front-end for the HEIR
    ecosystem of dialects. It is supposed to be fully interoperable with the
    rest of MLIR via secret.generic, while lower-level HEIR dialects would have
    custom types for arithmetic on secret integers of various bit widths.
  }];

  let cppNamespace = "::mlir::heir::secret";
  let useDefaultTypePrinterParser = 1;
}

#endif  // HEIR_INCLUDE_DIALECT_SECRET_IR_SECRETDIALECT_TD_
