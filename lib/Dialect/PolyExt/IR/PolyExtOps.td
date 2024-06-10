#ifndef HEIR_LIB_DIALECT_POLYEXT_IR_POLYEXTOPS_TD_
#define HEIR_LIB_DIALECT_POLYEXT_IR_POLYEXTOPS_TD_

include "PolyExtDialect.td"
include "mlir/Dialect/Polynomial/IR/PolynomialTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


class PolyExt_Op<string mnemonic, list<Trait> traits = []> :
        Op<PolyExt_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::heir::poly_ext";
}

def PolyExt_CModSwitchOp : PolyExt_Op<"cmod_switch", traits = [Pure, ElementwiseMappable]> {
  let summary = "Move polynomial into a ring with different `cmod` by scaling polynomial coefficients proportionally";

  let description = [{
    `poly_ext.cmod_switch` "moves" polynomial $x$ from ring $Z/QZ[X]/(f(x))$ to the ring $Z/qZ[X]/(f(x))$.
    The move is done by scaling the polynomial coefficients by $q/Q$ and "rounding" so input
    and output coefficients are congruent modulo `congruence_modulus` $t$.
    The resulting polynomial $\tilde{x} \approx (q/Q)x$ and $\tilde{x} \equiv x \mod t$.
  }];
  let arguments = (ins
    PolynomialLike:$x,
    // TODO: make congruence_modulus optional with default value 1
    Builtin_IntegerAttr:$congruence_modulus
  );

  let results = (outs
    PolynomialLike:$output
  );

  let assemblyFormat = "$x attr-dict `:` type($x) `->` type($output)";

  let hasVerifier = 1;
}


#endif  // HEIR_LIB_DIALECT_POLYEXT_IR_POLYEXTOPS_TD_
