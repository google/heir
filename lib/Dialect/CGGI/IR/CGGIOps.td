#ifndef LIB_DIALECT_CGGI_IR_CGGIOPS_TD_
#define LIB_DIALECT_CGGI_IR_CGGIOPS_TD_

include "lib/Dialect/HEIRInterfaces.td"
include "lib/Dialect/CGGI/IR/CGGIDialect.td"
include "lib/Dialect/CGGI/IR/CGGIAttributes.td"

include "mlir/Dialect/Polynomial/IR/PolynomialAttributes.td"
include "lib/Dialect/LWE/IR/LWETypes.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonAttrConstraints.td"
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

def CGGI_PackedOp : CGGI_Op<"packed_gates", [
    Pure,
    SameOperandsAndResultType
]> {
  let arguments = (ins
    CGGIGate_Attr:$gates,
    LWECiphertextLike:$lhs,
    LWECiphertextLike:$rhs
  );

  let results = (outs LWECiphertextLike:$output);
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
  let hasCanonicalizeMethod = 1;
}

def CGGI_Lut3Op : CGGI_LutOp<"lut3", [AllTypesMatch<["a", "b", "c", "output"]>]> {
  let summary = "A lookup table on three inputs.";
  let arguments = (ins LWECiphertextLike:$c, LWECiphertextLike:$b, LWECiphertextLike:$a, Builtin_IntegerAttr:$lookup_table);
  let results = (outs LWECiphertextLike:$output);
  let hasCanonicalizeMethod = 1;
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

  // Use Variadic<AnyType> and enforce LWECiphertextLike on type in verifier.
  // Tablegen produces an error when validating types when using a TypeOrContainer
  // type in a Variadic like Variadic<LWECiphertextLike>:
  // A workaround uses Variadic<AnyType> and a custom type verifier.
  // error: Value specified for template argument 'Variadic:type' is of type
  //   TypeOrContainer; expected type Type: LWECiphertextLike
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    DenseI32ArrayAttr:$coefficients,
    Builtin_IntegerAttr:$lookup_table
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = "operands attr-dict `:` type($output)" ;
  let hasVerifier = 1;
}

def CGGI_MultiLutLinCombOp : CGGI_Op<"multi_lut_lincomb", [
  Pure,
  Commutative,
  ElementwiseMappable,
  Scalarizable
]> {
  let summary = "A multi-output version of lut_lincomb with one LUT per output.";
  let description = [{
    An op representing multiple lookup tables applied to a shared input, which
    is prepared via a static linear combination. This is equivalent to
    `cggi.lut_lincomb`, but where the linear combination is given to multiple
    lookup tables, each producing a separate output.

    This can be achieved by a special implementation of blind rotate in the CGGI
    scheme. See [AutoHoG](https://ieeexplore.ieee.org/document/10413195).

    Example:

    ```mlir
    #encoding = #lwe.bit_field_encoding<cleartext_start=30, cleartext_bitwidth=3>
    #params = #lwe.lwe_params<cmod=7917, dimension=4>
    !ciphertext = !lwe.lwe_ciphertext<encoding = #encoding, lwe_params = #params>

    %4 = cggi.multi_lut_lincomb %c0, %c1, %c2, %c3 {
        coefficients = array<i32: 1, 2, 3, 2>,
        lookup_tables = array<index: 68, 70, 4, 8>
    } : (!ciphertext, !ciphertext, !ciphertext, !ciphertext) -> (!ciphertext, !ciphertext, !ciphertext, !ciphertext)
    ```

    Represents applying the following LUTs. Performance-wise, this is
    comparable to applying a single LUT to a linear combination.

    ```
    x = (1 * c0 + 2 * c1 + 3 * c2 + 2 * c3)
    return (
      (68 >> x) & 1,
      (70 >> x) & 1,
      (4 >> x) & 1,
      (8 >> x) & 1
    )
    ```
  }];

  let arguments = (ins
    Variadic<LWECiphertext>:$inputs,
    DenseI32ArrayAttr:$coefficients,
    DenseI32ArrayAttr:$lookup_tables
  );
  let results = (outs Variadic<LWECiphertext>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type($inputs, $outputs)" ;
  let hasVerifier = 1;
}

#endif  // LIB_DIALECT_CGGI_IR_CGGIOPS_TD_
