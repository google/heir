#ifndef HEIR_INCLUDE_DIALECT_POLY_IR_POLYOPS_TD_
#define HEIR_INCLUDE_DIALECT_POLY_IR_POLYOPS_TD_

include "PolyAttributes.td"
include "PolyDialect.td"
include "PolyTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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

class Poly_BinOp<string mnemonic, list<Trait> traits = [Pure, SameOperandsAndResultType, Commutative]> :
        Poly_Op<mnemonic, traits> {
  let arguments = (ins Polynomial:$lhs, Polynomial:$rhs);
  let results = (outs Polynomial:$output);
  let assemblyFormat = "`(` operands `)` attr-dict `:` type($output)" ;
}

def Poly_AddOp : Poly_BinOp<"add"> {
  let summary = "Addition operation between polynomials.";
}

def Poly_SubOp : Poly_BinOp<"sub"> {
  let summary = "Subtraction operation between polynomials.";
}

def Poly_MulOp : Poly_BinOp<"mul"> {
  let summary = "Multiplication operation between polynomials.";
}

def Poly_MulConstantOp : Poly_Op<"mul_constant"> {
  let summary = "Multiplication by a constant of the field.";

  let arguments = (ins
    Polynomial:$x,
    AnyInteger:$constant
  );

  let results = (outs
    Polynomial:$output
  );
}

//===----------------------------------------------------------------------===//
// Poly extraction operations.
//===----------------------------------------------------------------------===//

def Poly_GetCoeffOp : Poly_Op<"get_coeff"> {
  let summary = "Gets the coefficient of X^{index} from a polynomial.";

  let arguments = (ins
    Polynomial:$x,
    Index:$index
  );

  // NOTE: This assumes the underlying type is an integer.
  let results = (outs
    AnyInteger:$output
  );
}

def Poly_ExtractSliceOp : Poly_Op<"extract_slice", [Pure, InferTypeOpAdaptor]> {
  let summary = "Extracts a slice of coefficients from X^i to X^j.";

  let description = [{
    Extracts a slice of coefficients from X^i to X^j.

    The included range includes `i`, but excludes `j`, similar to array slice
    semantics.

    The coefficients are returned in the order they appear in the polynomial,
    from smallest degree to largest degree

  }];
  let arguments = (ins Polynomial:$input, Index:$i, Index:$j);
  let results = (outs TensorOf<[AnyInteger]>:$output);
}

//===----------------------------------------------------------------------===//
// Poly generation operations.
//===----------------------------------------------------------------------===//

def Poly_PolyFromCoeffsOp : Poly_Op<"from_coeffs", [Pure]> {
  let summary = "Creates a Polynomial from integer coefficients stored in a tensor.";
  let arguments = (ins TensorOf<[AnyInteger]>:$input);
  let results = (outs Polynomial:$output);

  let builders = [
    // Builder that infers coefficient modulus from tensor bit width,
    // and uses whatever input ring is provided by the caller.
    OpBuilder<(ins "::mlir::Value":$input, "RingAttr":$ring)>
  ];
}

#endif  // HEIR_INCLUDE_DIALECT_POLY_IR_POLYOPS_TD_
