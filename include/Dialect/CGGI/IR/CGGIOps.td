#ifndef HEIR_INCLUDE_DIALECT_CGGI_IR_CGGIOPS_TD_
#define HEIR_INCLUDE_DIALECT_CGGI_IR_CGGIOPS_TD_

include "include/Dialect/HEIRInterfaces.td"
include "include/Dialect/CGGI/IR/CGGIDialect.td"

include "include/Dialect/Polynomial/IR/PolynomialAttributes.td"
include "include/Dialect/LWE/IR/LWETypes.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class CGGI_Op<string mnemonic, list<Trait> traits = []> :
        Op<CGGI_Dialect, mnemonic, traits> {
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`  `(` qualified(type(operands)) `)` `->` qualified(type(results))
  }];
  let cppNamespace = "::mlir::heir::cggi";
}

// --- Operations for a gate-bootstrapping API of a CGGI library ---
def LWECiphertextLike : TypeOrContainer<LWECiphertext, "ciphertext-like">;

class CGGI_BinaryGateOp<string mnemonic>
  : CGGI_Op<mnemonic, [
    Pure,
    Commutative,
    SameOperandsAndResultType,
    ElementwiseMappable,
    Scalarizable
]> {
  let arguments = (ins LWECiphertextLike:$lhs, LWECiphertextLike:$rhs);
  let results = (outs LWECiphertextLike:$output);
  // Note: error: type of result #0, named 'output', is not buildable and a buildable type cannot be inferred
  // LWECiphertext is not buildable?
  let assemblyFormat = "operands attr-dict `:` qualified(type($output))" ;
}

def CGGI_AndOp : CGGI_BinaryGateOp<"and"> { let summary = "Logical AND of two ciphertexts."; }
def CGGI_NandOp : CGGI_BinaryGateOp<"nand"> { let summary = "Logical NAND of two ciphertexts."; }
def CGGI_NorOp  : CGGI_BinaryGateOp<"nor">  { let summary = "Logical NOR of two ciphertexts."; }
def CGGI_OrOp  : CGGI_BinaryGateOp<"or">  { let summary = "Logical OR of two ciphertexts."; }
def CGGI_XorOp : CGGI_BinaryGateOp<"xor"> { let summary = "Logical XOR of two ciphertexts."; }
def CGGI_XNorOp : CGGI_BinaryGateOp<"xnor"> { let summary = "Logical XNOR of two ciphertexts."; }

def CGGI_NotOp : CGGI_Op<"not", [Pure, SameOperandsAndResultType, Involution]> {
  let arguments = (ins LWECiphertext:$input);
  let results = (outs LWECiphertext:$output);
  let assemblyFormat = "operands attr-dict `:` qualified(type($output))";
  let summary = "Logical NOT of two ciphertexts";
}

class CGGI_LutOp<string mnemonic, list<Trait> traits = []>
  : CGGI_Op<mnemonic, traits # [
  Pure,
  Commutative,
  ElementwiseMappable,
  Scalarizable,
  DeclareOpInterfaceMethods<LUTOpInterface>

]> {
  let results = (outs LWECiphertextLike:$output);
  let assemblyFormat = "`(` operands `)` attr-dict `:` qualified(type($output))" ;

  let description = [{
    An op representing a lookup table applied to some number `n` of ciphertexts
    encrypting boolean input bits.

    Over cleartext bits `a, b, c`, using `n = 3` for example, the operation
    computed by this function can be interpreted as

    ```
      truth_table >> {c, b, a}
    ```

    where `{c, b, a}` is the unsigned 3-bit integer with bits `c, b, a` from most
    significant bit to least-significant bit. The input are combined into a
    single ciphertext input to the lookup table using products with plaintexts
    and sums.
  }];
}

def CGGI_Lut2Op : CGGI_LutOp<"lut2", [AllTypesMatch<["a", "b", "output"]>]> {
  let summary = "A lookup table on two inputs.";
  let arguments = (ins LWECiphertextLike:$b, LWECiphertextLike:$a, Builtin_IntegerAttr:$lookup_table);
}

def CGGI_Lut3Op : CGGI_LutOp<"lut3", [AllTypesMatch<["a", "b", "c", "output"]>]> {
  let summary = "A lookup table on three inputs.";
  let arguments = (ins LWECiphertextLike:$c, LWECiphertextLike:$b, LWECiphertextLike:$a, Builtin_IntegerAttr:$lookup_table);
  let results = (outs LWECiphertextLike:$output);
}

#endif  // HEIR_INCLUDE_DIALECT_CGGI_IR_CGGIOPS_TD_
