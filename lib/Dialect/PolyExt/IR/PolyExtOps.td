#ifndef HEIR_LIB_DIALECT_POLYEXT_IR_POLYEXTOPS_TD_
#define HEIR_LIB_DIALECT_POLYEXT_IR_POLYEXTOPS_TD_

include "PolyExtDialect.td"
include "lib/Dialect/Polynomial/IR/PolynomialAttributes.td"
include "lib/Dialect/Polynomial/IR/PolynomialDialect.td"
include "lib/Dialect/Polynomial/IR/PolynomialTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


class PolyExt_Op<string mnemonic, list<Trait> traits = []> :
        Op<PolyExt_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::heir::poly_ext";
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
}

def PolyExt_CModSwitchOp : PolyExt_Op<"cmod_switch", traits = [Pure, ElementwiseMappable]> {
  let summary = "Move polynomial into a ring with different `cmod` by scaling polynomial coefficients proportionally";

  let description = [{
    `poly_ext.cmod_switch` "moves" polynomial $x$ from ring $Z/QZ[X]/(f(x))$ to the ring $Z/qZ[X]/(f(x))$.
    The move is done by scaling the polynomial coefficients by $q/Q$ and "rounding" so input and output coufficients are congruent modulo `congruence_modulus` $t$.
    The resulting polynomial $\tilde{x} \approx (q/Q)x$ and $\tilde{x} \equiv x \mod t$.
  }];
  let arguments = (ins
    PolynomialLike:$x,
    // TODO: make congruence_modulus optional with default value 1
    Builtin_IntegerAttr:$congruence_modulus
  );
  let results = (outs PolynomialLike:$output);
  let hasVerifier = 1;
}

def PolyExt_EvalOp : PolyExt_Op<"eval", traits = [
    Pure, ElementwiseMappable, AllTypesMatch<["input", "output"]>
]> {
  let summary = "Evaluate a polynomial on an input not from the polynomial's ring";
  let description = [{
    `poly_ext.eval` evaluates a statically-known polynomial on an arbitrary
    input not a member of the polynomial's underlying ring.

    For example, this can be used to represent a polynomial function of a matrix
    or ciphertext input. In HEIR this is used to separate the two steps of
    selecting a polynomial approximation for a given operation (e.g., Remez or
    related algorithms) from the optimizations involved in evaluating that
    polynomial (e.g., Paterson-Stockmeyer vs Baby Step Giant Step).
  }];
  let arguments = (ins
    AnyType:$input,
    Polynomial_PolynomialAttr:$polynomial
  );
  let results = (outs AnyType:$output);
}



#endif  // HEIR_LIB_DIALECT_POLYEXT_IR_POLYEXTOPS_TD_
