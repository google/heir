#ifndef HEIR_LIB_DIALECT_CGGI_IR_CGGIOPS_TD_
#define HEIR_LIB_DIALECT_CGGI_IR_CGGIOPS_TD_

include "lib/Dialect/HEIRInterfaces.td"
include "lib/Dialect/CGGI/IR/CGGIDialect.td"

include "mlir/Dialect/Polynomial/IR/PolynomialAttributes.td"
include "lib/Dialect/LWE/IR/LWETypes.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class CGGI_Op<string mnemonic, list<Trait> traits = []> :
        Op<CGGI_Dialect, mnemonic, traits> {
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
  let cppNamespace = "::mlir::heir::cggi";
}

// --- Operations for a gate-bootstrapping API of a CGGI library ---

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

def CGGI_NotOp : CGGI_Op<"not", [
    Pure,
    Involution,
    SameOperandsAndResultType,
    ElementwiseMappable,
    Scalarizable
]> {
  let arguments = (ins LWECiphertextLike:$input);
  let results = (outs LWECiphertextLike:$output);
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
  let assemblyFormat = "operands attr-dict `:` qualified(type($output))" ;

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

def CGGI_LutLinCombOp : CGGI_Op<"lut_lincomb", [
  Pure,
  Commutative,
  ElementwiseMappable,
  Scalarizable,
  SameOperandsAndResultType,
  DeclareOpInterfaceMethods<LUTOpInterface>
]> {
  let summary = "A variadic-input lookup table with inputs prepared via linear combination.";
  let description = [{
    An op representing a lookup table applied to an arbitrary number of
    input ciphertexts, which are combined according to a static linear
    combination attached to the op.

    The user must ensure the chosen linear combination does not bleed error
    bits into the message space according to the underlying ciphertext's
    encoding attributes. E.g., a bit_field_encoding with 3 cleartext bits
    cannot be multiplied by 16.

    Example:

    ```mlir
    #encoding = #lwe.bit_field_encoding<cleartext_start=30, cleartext_bitwidth=3>
    #params = #lwe.lwe_params<cmod=7917, dimension=4>
    !ciphertext = !lwe.lwe_ciphertext<encoding = #encoding, lwe_params = #params>

    %4 = cggi.lut_lincomb %c0, %c1, %c2, %c3 {coefficients = array<i32: 1, 2, 3, 2>, lookup_table = 68 : index} : !ciphertext
    ```

    Represents applying the lut

    ```
      68 >> (1 * c0 + 2 * c1 + 3 * c2 + 2 * c3)
    ```
  }];

  let arguments = (ins
    Variadic<LWECiphertext>:$inputs,
    DenseI32ArrayAttr:$coefficients,
    Builtin_IntegerAttr:$lookup_table
  );
  let results = (outs LWECiphertext:$output);
  let assemblyFormat = "operands attr-dict `:` type($output)" ;
  let hasVerifier = 1;
}

#endif  // HEIR_LIB_DIALECT_CGGI_IR_CGGIOPS_TD_
