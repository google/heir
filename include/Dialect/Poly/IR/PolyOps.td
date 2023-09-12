#ifndef HEIR_INCLUDE_DIALECT_POLY_IR_POLYOPS_TD_
#define HEIR_INCLUDE_DIALECT_POLY_IR_POLYOPS_TD_

include "PolyAttributes.td"
include "PolyDialect.td"
include "PolyTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Poly_Op<string mnemonic, list<Trait> traits = []> :
        Op<Poly_Dialect, mnemonic, traits> {
  // See https://mlir.llvm.org/docs/DefiningDialects/Operations/#declarative-assembly-format
  // TODO(b/291601096): Simplify this format by adding type constraints.
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`  `(` type(operands) `)` `->` type(results)
  }];
  let cppNamespace = "::mlir::heir::poly";
}

class Poly_BinOp<string mnemonic, list<Trait> traits = [Pure, SameOperandsAndResultType, Commutative, ElementwiseMappable]> :
        Poly_Op<mnemonic, traits> {
  let arguments = (ins PolynomialLike:$lhs, PolynomialLike:$rhs);
  let results = (outs PolynomialLike:$output);
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

def Poly_MulConstantOp : Poly_Op<"mul_constant", [ElementwiseMappable, AllTypesMatch<["x", "output"]>]> {
  let summary = "Multiplication by a constant of the field.";

  let arguments = (ins
    PolynomialLike:$x,
    AnyInteger:$constant
  );

  let results = (outs
    PolynomialLike:$output
  );

  let assemblyFormat = "`(` operands `)` attr-dict `:` type($x) `,` type($constant)" ;
}

def Poly_FromTensorOp : Poly_Op<"from_tensor", [Pure]> {
  let summary = "Creates a polynomial from integer coefficients stored in a tensor.";
  let description = [{
    `poly.from_tensor` creates a polynomial value from a tensor of coefficients.
    The input tensor must list the coefficients in degree-increasing order.

    The input one-dimensional tensor may have size at most the degree of the
    ring's ideal generator polynomial, with smaller dimension implying that
    all higher-degree terms have coefficient zero.
  }];
  let arguments = (ins RankedTensorOf<[AnyInteger]>:$input);
  let results = (outs Polynomial:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` qualified(type($output))";

  let builders = [
    // Builder that infers coefficient modulus from tensor bit width,
    // and uses whatever input ring is provided by the caller.
    OpBuilder<(ins "::mlir::Value":$input, "RingAttr":$ring)>
  ];

  let hasVerifier = 1;
}

def Poly_ToTensorOp : Poly_Op<"to_tensor", [Pure]> {
  let summary = "Creates a tensor containing the coefficients of a polynomial.";
  let description = [{
    `poly.to_tensor` creates a tensor value containing the coefficients of the
    input polynomial. The output tensor contains the coefficients in
    degree-increasing order.

    Operations that act on the coefficients of a polynomial, such as extracting
    a specific coefficient or extracting a range of coefficients, should be
    implemented by composing `to_tensor` with the relevant `tensor` dialect
    ops.

    The output tensor has shape equal to the degree of the ring's ideal
    generator polynomial, including zeroes.
  }];
  let arguments = (ins Polynomial:$input);
  let results = (outs RankedTensorOf<[AnyInteger]>:$output);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `->` type($output)";

  let hasVerifier = 1;
}

#endif  // HEIR_INCLUDE_DIALECT_POLY_IR_POLYOPS_TD_
