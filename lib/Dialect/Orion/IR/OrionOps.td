#ifndef LIB_DIALECT_ORION_IR_ORIONOPS_TD_
#define LIB_DIALECT_ORION_IR_ORIONOPS_TD_

include "lib/Dialect/Orion/IR/OrionDialect.td"
include "lib/Dialect/LWE/IR/LWETypes.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Orion_Op<string mnemonic, list<Trait> traits = []> :
        Op<Orion_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::heir::orion";
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)" ;
}

def Orion_LinearTransformOp : Orion_Op<"linear_transform", [Pure]> {
  // Use a cleartext for the diagonals since they need to be rotated in the
  // BSGS implementation of linear transforms.
  let arguments = (ins LWECiphertext:$input, FloatLike:$diagonals);
  let results = (outs LWECiphertext:$result);
  let summary = "Apply a linear transformation to the input ciphertext using pre-computed diagonals";
}

def Orion_ChebyshevOp : Orion_Op<"chebyshev", [Pure]> {
  let arguments = (ins
    LWECiphertext:$input,
    ArrayAttr:$coefficients,
    Builtin_FloatAttr:$domain_start,
    Builtin_FloatAttr:$domain_end
  );
  let results = (outs LWECiphertext:$result);
  let summary = "Evaluates a Chebyshev polynomial on a ciphertext using pre-computed coefficients";
}

#endif  // LIB_DIALECT_ORION_IR_ORIONOPS_TD_
