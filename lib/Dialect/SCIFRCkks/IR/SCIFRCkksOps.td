//===- SCIFRCkksOps.td - SCIFRCkks Dialect ops -----------*- tablegen -*-===//

#ifndef SCIFRCkks_OPS
#define SCIFRCkks_OPS

include "SCIFRCkksDialect.td"
include "mlir/IR/AttrTypeBase.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"

include "lib/Dialect/HEIRInterfaces.td"

include "lib/Dialect/LWE/IR/LWETypes.td"

class SCIFRCkks_Op<string mnemonic, list<Trait> traits = []> :
        Op<SCIFRCkks_Dialect, mnemonic, traits> {
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
  let cppNamespace = "::mlir::scifrckks";
}

// --- Operations for a gate-bootstrapping API of a SCIFRCkks library ---

class SCIFRCkks_BinaryGateOp<string mnemonic>
  : SCIFRCkks_Op<mnemonic, [
    Pure,
    Commutative,
    SameOperandsAndResultType,
    ElementwiseMappable,
    Scalarizable
]> {
  let arguments = (ins LWECiphertext:$lhs, LWECiphertext:$rhs);
  let results = (outs LWECiphertext:$output);
  // Note: error: type of result #0, named 'output', is not buildable and a buildable type cannot be inferred
  // LWECiphertext is not buildable?
  let assemblyFormat = "operands attr-dict `:` qualified(type($output))" ;
}

def SCIFRCkks_AddOp : SCIFRCkks_BinaryGateOp<"Add"> { let summary = "Add two ciphertexts."; }
def SCIFRCkks_SubOp : SCIFRCkks_BinaryGateOp<"Sub"> { let summary = "Sub two ciphertexts."; }
def SCIFRCkks_MulOp : SCIFRCkks_BinaryGateOp<"Mul"> { let summary = "Mul two ciphertexts."; }
def SCIFRCkks_AddConstOp : SCIFRCkks_BinaryGateOp<"AddC"> { let summary = "Add ciphertext with constant Op."; }
def SCIFRCkks_SubConstOp  : SCIFRCkks_BinaryGateOp<"SubC">  { let summary = "Subtract ciphertext from a constant Op."; }
def SCIFRCkks_RelinOp  : SCIFRCkks_Op<"Relin">  { let summary = "Relinearize a ciphertext."; }

//===----------------------------------------------------------------------===//
def SCIFRCkks_AnyNumber : AnyTypeOf<[AnyFloat],
                               "number">;

/// SCIFRCKKS Scheme ops in ciphertext ///

#endif // SCIFRCkks_Ops
