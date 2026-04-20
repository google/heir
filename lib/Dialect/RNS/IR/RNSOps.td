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

def RNS_ExtractSingleSliceOp : RNS_Op<"extract_single_slice", [Pure, ElementwiseMappable, DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>]> {
  let summary = "Extract one RNS limb as the type of the limb";
  let description = [{
    Given an RNS-typed value, extract the basis limb at `index` and return it
    as the corresponding type.

    For shaped inputs, this operation is applied elementwise, preserving the
    input container shape while replacing each RNS element with the selected
    limb.
  }];
  let arguments = (ins RNSLike:$input, IndexAttr:$index);
  let results = (outs AnyType:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let hasVerifier = 1;
}

def RNS_PackOp : RNS_Op<"pack", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>]> {
  let summary = "Pack multiple limb values into one RNS value";
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
  let arguments = (ins Variadic<AnyType>:$input);
  let results = (outs RNS:$output);
  let assemblyFormat = "operands attr-dict `:` type($input)";
}


#endif  // LIB_DIALECT_RNS_IR_RNSOPS_TD_
