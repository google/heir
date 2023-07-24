#ifndef HEIR_INCLUDE_DIALECT_POLY_IR_POLYOPS_TD_
#define HEIR_INCLUDE_DIALECT_POLY_IR_POLYOPS_TD_

include "PolyDialect.td"
include "PolyTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// Poly Operation definitions.
//===----------------------------------------------------------------------===//

class Poly_Op<string mnemonic, list<Trait> traits = []> :
        Op<Poly_Dialect, mnemonic, traits> {

  // See https://mlir.llvm.org/docs/DefiningDialects/Operations/#declarative-assembly-format
  // TODO(b/291601096): Simplify this format by adding type constraints.
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`  `(` type(operands) `)` `->` type(results)
  }];
  let cppNamespace = "::mlir::heir::poly";
}

//===----------------------------------------------------------------------===//
// Ring operations.
//===----------------------------------------------------------------------===//

def Poly_AddOp : Poly_Op<"add", [SameOperandsAndResultType]> {
  let summary = "Addition operation between polynomials.";

  let arguments = (ins
    Variadic<Poly>:$x
  );

  let results = (outs
    Poly:$output
  );

   let assemblyFormat = "`(` operands `)` attr-dict `:` type($output)" ;
}

def Poly_SubOp : Poly_Op<"sub", [SameOperandsAndResultType]> {
  let summary = "Subtraction operation between polynomials.";

  let arguments = (ins
    Variadic<Poly>:$x
  );

  let results = (outs
    Poly:$output
  );

  let assemblyFormat = "`(` operands `)` attr-dict `:` type($output)" ;
}

def Poly_MulOp : Poly_Op<"mul", [SameOperandsAndResultType]> {
  let summary = "Multiplication operation between polynomials.";

  let arguments = (ins
    Variadic<Poly>:$x
  );

  let results = (outs
    Poly:$output
  );

  let assemblyFormat = "`(` operands `)` attr-dict `:` type($output)" ;
}

def Poly_MulConstantOp : Poly_Op<"mul_constant"> {
  let summary = "Multiplication by a constant of the field.";

  let arguments = (ins
    Poly:$x,
    AnyInteger:$constant
  );

  let results = (outs
    Poly:$output
  );
}

//===----------------------------------------------------------------------===//
// Poly extraction operations.
//===----------------------------------------------------------------------===//

def Poly_GetCoeffOp : Poly_Op<"get_coeff"> {
  let summary = "Gets the coefficient of X^{index} from a polynomial.";

  let arguments = (ins
    Poly:$x,
    Index:$index
  );

  // NOTE: This assumes the underlying type is an integer.
  let results = (outs
    AnyInteger:$output
  );
}

def Poly_ExtractSliceOp : Poly_Op<"extract_slice"> {
  let summary = "Extracts a slice of coefficients from X^i to X^j inclusive.";

  let arguments = (ins
    Poly:$x,
    Index:$i,
    Index:$j
  );

  let results = (outs
    TensorOf<[AnyInteger]>:$output
  );
}

//===----------------------------------------------------------------------===//
// Poly generation operations.
//===----------------------------------------------------------------------===//

def Poly_PolyFromCoeffsOp : Poly_Op<"from_coeffs"> {
  let summary = "Creates a Polynomial from integer coeffs.";

  let arguments = (ins
    TensorOf<[AnyInteger]>:$x
  );

  let results = (outs
    Poly:$output
  );
}

#endif  // HEIR_INCLUDE_DIALECT_POLY_IR_POLYOPS_TD_
