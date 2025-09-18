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

def RNS_ConvertBasisOp : RNS_Op<"convert_basis", [Pure, ElementwiseMappable]> {
  let summary = "Extend the basis of an RNS value to a different basis";
  let description = [{
  TODO: fill in
  }];
  let arguments = (ins RNSLike:$input);
  let results = (outs RNSLike:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";

  // FIXME: verifier that all RNS component moduli are odd
}

#endif  // LIB_DIALECT_RNS_IR_RNSOPS_TD_
