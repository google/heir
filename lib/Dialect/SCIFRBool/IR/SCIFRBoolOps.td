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
  let cppNamespace = "::mlir::cornami::scifrbool";
}

// --- Operations for a gate-bootstrapping API of a SCIFRBool library ---
class SCIFRBool_BinaryGateOp<string mnemonic>
  : SCIFRBool_Op<mnemonic, [
    Pure,
    Commutative,
    SameOperandsAndResultType
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
]> {
  let arguments = (ins LWECiphertext:$input);
  let results = (outs LWECiphertext:$output);
  let assemblyFormat = "operands attr-dict `:` qualified(type($output))";
  let summary = "Logical NOT of two ciphertexts";
}

//===----------------------------------------------------------------------===//
def SCIFRBOOL_AnyNumber : AnyTypeOf<[AnyFloat],
                               "number">;


// Operator for kernel outlining pass; this pass will contain all the ops
// that are able to be scheduled on the currently available hardware resources
def SCIFRBOOL_SectionOp : SCIFRBool_Op<"section",[NoTerminator]> {
  let summary = "Wraps a single operation";
  let description = [{
      Section op is intended to handle kernel outlining, move kernels of instructions. Section contains the code scheduled to Fractal Array.
  }];

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
