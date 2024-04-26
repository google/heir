#ifndef LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALDIALECT_TD_
#define LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// Polynomial dialect definitions
//===----------------------------------------------------------------------===//

def Polynomial_Dialect : Dialect {
  // The namespace of the dialect.
  // This corresponds to the string provided in `getDialectNamespace`.
  let name = "_polynomial";

  let description = [{
    The Polynomial dialect defines single-variable polynomial types and
    operations.

    The simplest use of this dialect is to do math in a polynomial ring `R[x]`,
    where `R` is another MLIR integer type like `u32`, and lower to `arith`.

    More generally, this dialect represent polynomial operations in a quotient
    polynomial ring `Z/qZ[X]/(f(x))` for some integer `q` and polynomial `f(x)`.
    Polyomials `p(x), q(x)` are equal in this ring if they have the same remainder
    when dividing by `f(x)`. The canonical representative for a polynomial `p(x)`
    is has degree less than `deg(f(x))`. When a modulus is given, ring operations
    are polynomial addition and multiplication performed with reductions modulo
    `f(x)` and `q`.
  }];

  let cppNamespace = "::mlir::heir::polynomial";

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif  // LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALDIALECT_TD_
