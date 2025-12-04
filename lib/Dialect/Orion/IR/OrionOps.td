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
  let arguments = (ins
    LWECiphertext:$input,
    FloatLike:$diagonals,
    DenseI32ArrayAttr:$diagonal_indices,
    Builtin_IntegerAttr:$orion_level,
    Builtin_FloatAttr:$bsgs_ratio,
    Builtin_IntegerAttr:$slots
  );
  let results = (outs LWECiphertext:$result);
  let summary = "Apply a linear transformation to the input ciphertext using pre-computed diagonals";
  let description = [{
    This operation applies a linear transformation on a ciphertext using
    the provided float diagonals.

    The `diagonals` input is a tensor where each row represents one non-zero
    diagonal of the square matrix to evaluate.

    The `diagonal_indices` attribute specifies the index of each corresponding diagonal
    in `diagonals`. I.e., the first diagonal in the original matrix may have been zero
    and omitted, and as a result the first entry of `diagonals` corresponds to
    the diagonal with index `diagonal_indices[0]`, and so on.

    The `orion_level` attribute specifies the modulus level at which the operation
    should be performed.

    The `bsgs_ratio` attribute is used to optimize the linear transformation
    using the baby-step giant-step algorithm.

    The `slots` attribute specifies the number of slots in the ciphertext.
  }];
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
