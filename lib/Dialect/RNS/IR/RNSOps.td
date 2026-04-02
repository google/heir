#ifndef LIB_DIALECT_RNS_IR_RNSOPS_TD_
#define LIB_DIALECT_RNS_IR_RNSOPS_TD_

include "lib/Dialect/RNS/IR/RNSDialect.td"
include "lib/Dialect/RNS/IR/RNSTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class RNS_Op<string mnemonic, list<Trait> traits = []> :
        Op<RNS_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::heir::rns";
}

def RNS_ExtractSliceOp : RNS_Op<"extract_slice", [Pure, ElementwiseMappable, DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>]> {
  let summary = "Extracts a slice of RNS limbs";
  let description = [{
    Given an RNS-typed value with $k$ basis types (limbs), extract the slice of
    RNS components starting at `start` and having size `size`.

    The result type is an RNS type containing the subset of basis types
    corresponding to the extracted slice. This is useful for operations like
    truncating or partitioning a modulus chain.
  }];
  let arguments = (ins RNSLike:$input, IndexAttr:$start, IndexAttr:$size);
  let results = (outs RNSLike:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let hasVerifier = 1;
}

def RNS_ExtractResidueOp : RNS_Op<"extract_residue", [Pure, ElementwiseMappable, DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>]> {
  let summary = "Extract one RNS residue as the type of the limb";
  let description = [{
    Given an RNS-typed value, extract the basis limb at `index` and return it
    as the corresponding type.

    For shaped inputs, this operation is applied elementwise, preserving the
    input container shape while replacing each RNS element with the selected
    limb.
  }];
  let arguments = (ins RNSLike:$input, IndexAttr:$index);
  let results = (outs RNSBasisTypeInterfaceLike:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let hasVerifier = 1;
}

def RNS_PackOp : RNS_Op<"pack", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>]> {
  let summary = "Pack multiple residues into one RNS value";
  let description = [{
    `rns.pack` converts a list of `mod_arith` values into a single value with
    RNS type.

    The output RNS basis is formed by the input operand types in operand order,
    so operand `i` becomes limb `i` of the packed result.

    Examples:
    ```
    !zp17 = !mod_arith.int<17 : i32>
    !zp257 = !mod_arith.int<257 : i32>
    !rns = !rns.rns<!zp17, !zp257>

    %packed = rns.pack %x, %y : !zp17, !zp257
    return %packed : !rns
    ```
  }];
  let arguments = (ins Variadic<RNSBasisTypeInterface>:$input);
  let results = (outs RNS:$output);
  let assemblyFormat = "operands attr-dict `:` type($input)";
}

def RNS_ConvertBasisOp : RNS_Op<"convert_basis", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>]> {
  let summary = "Change an RNS value's basis";
  let description = [{
    Given x \in Z_Q represented as {x_i mod q_i} where Q=\prod q_i,
    convertBasis returns the target-basis representation obtained by:
    1. Compute x's centered integer representative in [-Q/2, Q/2]
    2. Reduce the integer representation mod p_i for each p_i in the target basis.

    This performs the conversion directly from the RNS representation of x,
    without first reconstructing x over the integers. Note that both bases must
    be composed entirely of odd moduli; this algorithm does not work for bases
    with even moduli.
  }];
  let arguments = (ins
      RNS:$value,
      TypeAttr:$targetBasis
  );
  let results = (outs
    RNS:$output
  );
  let assemblyFormat = "$value attr-dict `:` type($value) `->` type($output)";
}


#endif  // LIB_DIALECT_RNS_IR_RNSOPS_TD_
