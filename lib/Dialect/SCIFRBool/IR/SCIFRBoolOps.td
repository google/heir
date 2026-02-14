//===- SCIFRBoolOps.td - SCIFRBool Dialect ops -----------*- tablegen -*-===//

#ifndef SCIFRBool_OPS
#define SCIFRBool_OPS

include "SCIFRBoolDialect.td"
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

class SCIFRBool_Op<string mnemonic, list<Trait> traits = []> :
        Op<SCIFRBool_Dialect, mnemonic, traits> {
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
  let cppNamespace = "::mlir::scifrbool";
}

// --- Operations for a gate-bootstrapping API of a SCIFRBool library ---

class SCIFRBool_BinaryGateOp<string mnemonic>
  : SCIFRBool_Op<mnemonic, [
    Pure,
    Commutative,
    SameOperandsAndResultType,
    ElementwiseMappable,
    Scalarizable
]> {
  let arguments = (ins LWECiphertext:$lhs, LWECiphertext:$rhs);
  let results = (outs LWECiphertext:$output);
  let assemblyFormat = "operands attr-dict `:` qualified(type($output))" ;
}

def SCIFRBool_AndOp : SCIFRBool_BinaryGateOp<"and"> { let summary = "Logical AND of two ciphertexts."; }
def SCIFRBool_NandOp : SCIFRBool_BinaryGateOp<"nand"> { let summary = "Logical NAND of two ciphertexts."; }
def SCIFRBool_NorOp  : SCIFRBool_BinaryGateOp<"nor">  { let summary = "Logical NOR of two ciphertexts."; }
def SCIFRBool_OrOp  : SCIFRBool_BinaryGateOp<"or">  { let summary = "Logical OR of two ciphertexts."; }
def SCIFRBool_XorOp : SCIFRBool_BinaryGateOp<"xor"> { let summary = "Logical XOR of two ciphertexts."; }
def SCIFRBool_XNorOp : SCIFRBool_BinaryGateOp<"xnor"> { let summary = "Logical XNOR of two ciphertexts."; }

def SCIFRBool_NotOp : SCIFRBool_Op<"not", [
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
def SCIFRBOOL_AnyNumber : AnyTypeOf<[AnyFloat],
                               "number">;


def SCIFRBOOL_PBSOp : SCIFRBool_Op<"PBS", [SameOperandsAndResultType]> {
    let summary = "Programmatic Bootstrapping (PBS) operation.";
    let description = [{
        The `SCIFRBOOL.PBS` creates a PBS for all inputs.

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
def SCIFRBOOL_KSOp : SCIFRBool_Op<"KS",[SameOperandsAndResultType]> {
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
def SCIFRBOOL_LinearOp : SCIFRBool_Op<"Linear",[SameOperandsAndResultType]> {
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
/// boolean gates in ciphertext ///

def SCIFRBOOL_SectionOp : SCIFRBool_Op<"section",[NoTerminator]> {
  let summary = "Wraps a single operation";
  let arguments = (ins
    Variadic<AnyType>:$input
  );

  let results = (outs
    Variadic<AnyType>:$output
  );

  let regions = (region AnyRegion:$body);

  let assemblyFormat = "`(` operands `)` $body attr-dict `:` functional-type(operands, results)";
  let hasCustomAssemblyFormat = 1;
}

#endif // SCIFRBool_OpS
