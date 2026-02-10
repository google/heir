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
def SCIFRCkks_AddConstOp : SCIFRCkks_BinaryGateOp<"AddC"> { let summary = "Logical NAND of two ciphertexts."; }
def SCIFRCkks_SubConstOp  : SCIFRCkks_BinaryGateOp<"SubC">  { let summary = "Logical NOR of two ciphertexts."; }
def SCIFRCkks_RelinOp  : SCIFRCkks_BinaryGateOp<"or">  { let summary = "Logical OR of two ciphertexts."; }

def SCIFRCkks_NotOp : SCIFRCkks_Op<"not", [
    Pure,
    Involution,
    SameOperandsAndResultType,
    ElementwiseMappable,
    Scalarizable
]> {
  let arguments = (ins LWECiphertext:$input);
  let results = (outs LWECiphertext:$output);
  let assemblyFormat = "operands attr-dict `:` qualified(type($output))";
  let summary = "Logical NOT of two ciphertexts";
}

//===----------------------------------------------------------------------===//
def SCIFRCkks_AnyNumber : AnyTypeOf<[AnyFloat],
                               "number">;


def SCIFRCkks_PBSOp : SCIFRCkks_Op<"PBS", [SameOperandsAndResultType]> {
    let summary = "Programmatic Bootstrapping (PBS) operation.";
    let description = [{
        The `SCIFRCkks.PBS` creates a PBS for all inputs.

        This operation takes an tensor argument and returns a tensor.
    }];

    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

//===----------------------------------------------------------------------===//
// Operator: sigmoid
//===----------------------------------------------------------------------===//
def SCIFRCkks_KSOp : SCIFRCkks_Op<"KS",[SameOperandsAndResultType]> {
  let summary = "Computes Key Switching of input.";

  let description = [{
    KS : Key Switching of the operation
  }];

  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs
    AnyTensor:$output
  );
}

//===----------------------------------------------------------------------===//
// Operator: LinearOp
//===----------------------------------------------------------------------===//
def SCIFRCkks_LinearOp : SCIFRCkks_Op<"Linear",[SameOperandsAndResultType]> {
  let summary = "Computes linear combination of the input with fixed coefficients.";

  let description = [{
    compute the linear combination of the input FHE vectors using fixed coefficients
  }];

  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs
    AnyTensor:$output
  );
}
/// CKKS Scheme gates in ciphertext ///

#endif // SCIFRCkks_Ops
