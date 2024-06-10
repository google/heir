#ifndef LIB_DIALECT_RNS_IR_RNSDIALECT_TD_
#define LIB_DIALECT_RNS_IR_RNSDIALECT_TD_

include "mlir/IR/DialectBase.td"

def RNS_Dialect : Dialect {
  let name = "rns";
  let description = [{
    The `rns` dialect represents types and ops related to residue number
    system (RNS) representations of ring-like types, such as integers or
    polynomials decomposed from high-bit width to lower-bit-width prime
    moduli. Sometimes RNS is referred to as CRT, for "Chinese Remainder
    Theorem."

    This dialect is intended to be as generic as possible in terms of its
    interaction with standard MLIR. However, because of upstream MLIR
    constraints, we do not have the ability to override, say, `arith.addi`
    to operate on an `rns` type. So such situations require dedicated ops,
    canonicalization patterns, etc.
  }];

  let cppNamespace = "::mlir::heir::rns";

  let useDefaultTypePrinterParser = 1;

  let dependentDialects = [
    "mlir::polynomial::PolynomialDialect",
  ];
}

#endif  // LIB_DIALECT_RNS_IR_RNSDIALECT_TD_
