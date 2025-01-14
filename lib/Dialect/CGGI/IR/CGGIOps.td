#ifndef LIB_DIALECT_CGGI_IR_CGGIOPS_TD_
#define LIB_DIALECT_CGGI_IR_CGGIOPS_TD_

include "lib/Dialect/HEIRInterfaces.td"
include "lib/Dialect/CGGI/IR/CGGIDialect.td"
include "lib/Dialect/CGGI/IR/CGGIAttributes.td"

include "lib/Dialect/Polynomial/IR/PolynomialAttributes.td"
include "lib/Dialect/LWE/IR/LWETypes.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"
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

class CGGI_BinaryOp<string mnemonic>
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

def CGGI_AndOp : CGGI_BinaryOp<"and"> { let summary = "Logical AND of two ciphertexts."; }
def CGGI_NandOp : CGGI_BinaryOp<"nand"> { let summary = "Logical NAND of two ciphertexts."; }
def CGGI_NorOp  : CGGI_BinaryOp<"nor">  { let summary = "Logical NOR of two ciphertexts."; }
def CGGI_OrOp  : CGGI_BinaryOp<"or">  { let summary = "Logical OR of two ciphertexts."; }
def CGGI_XorOp : CGGI_BinaryOp<"xor"> { let summary = "Logical XOR of two ciphertexts."; }
def CGGI_XNorOp : CGGI_BinaryOp<"xnor"> { let summary = "Logical XNOR of two ciphertexts."; }
def CGGI_AddOp : CGGI_BinaryOp<"add"> { let summary = "Arithmetic addition of two ciphertexts."; }
def CGGI_MulOp : CGGI_BinaryOp<"mul"> {
  let summary = "Arithmetic multiplication of two ciphertexts.";
  let description = [{
    While CGGI does not have a native multiplication operation,
    some backend targets provide a multiplication
    operation that is implemented via a sequence
    of other atomic CGGI ops. When lowering to
    backends that do not have this, one must lower
    to this op the appropriate CGGI ops.
  }];
}

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

def CGGI_PackedOp : CGGI_Op<"packed_gates",
  [Pure, AllTypesMatch<["lhs", "rhs", "output"]>]>{
  let arguments = (ins
    CGGI_CGGIBoolGates:$gates,
    LWECiphertextLike:$lhs,
    LWECiphertextLike:$rhs
  );

      let description = [{
     Operation to where different Boolean gates are executed pairwise between elements of two ciphertext arrays.

     For example,

     ```mlir
     %0 = cggi.packed_gates %a, %b {gates = #cggi.cggi_gate<"and", "xor">} : tensor<2x!lwe.lwe_ciphertext>
     ```
     applies an "and" gate to the first elements of %a and %b and an xor gate to the
     second elements.

     Mapping is defined in the BooleanGates.td file.
  }];

  let results = (outs LWECiphertextLike:$output);
}

def CGGI_ProgrammableBootstrapOp : CGGI_Op<"programmable_bootstrap", [
    Pure,
    SameOperandsAndResultType,
    ElementwiseMappable,
    Scalarizable
]> {
  let arguments = (ins LWECiphertextLike:$input, Builtin_IntegerAttr:$lookup_table);
  let results = (outs LWECiphertextLike:$output);
  let assemblyFormat = "operands attr-dict `:` qualified(type($output))";
  let summary = "Programmable Bootstrap with a given lookup table.";

  let description = [{
    An op representing a programmable bootstrap applied to an LWE ciphertext.

    This operation evaluates a univariate function homomorphically on the
    ciphertext by selecting the correct value from a lookup table. The bit size
    of the lookup table integer attribute should be equal to the plaintext space
    size. For example, if there ciphertext can hold 3 plaintext message bits,
    then the lookup table must be represented at most by an integer with 8 bits.
  }];
  let hasVerifier = 1;
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

def CGGI_PackedLut3Op : CGGI_Op<"packed_lut3", [
    Pure,
    SameOperandsAndResultType,
    DeclareOpInterfaceMethods<LUTOpInterface>
]> {
  let arguments = (ins
    TypedArrayAttrBase<Builtin_IntegerAttr, "Array of integers">:$lookup_tables,
    LWECiphertextLike:$a,
    LWECiphertextLike:$b,
    LWECiphertextLike:$c
  );

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

def CGGI_SubOp : CGGI_Op<"sub", [
    Pure,
    SameOperandsAndResultType,
    ElementwiseMappable,
    Scalarizable
]> {
  let arguments = (ins LWECiphertextLike:$lhs, LWECiphertextLike:$rhs);
  let results = (outs LWECiphertextLike:$output);
  let assemblyFormat = "operands attr-dict `:` qualified(type($output))";
  let summary = "Subtraction of two ciphertexts.";
}


def CGGI_ShiftRightOp : CGGI_Op<"shr", [
    Pure,
]> {
  let arguments = (ins LWECiphertextLike:$lhs, AnyI8:$shiftAmount);
  let results = (outs LWECiphertextLike:$output);
  let summary = "Arithmetic shift to the right of a ciphertext by an integer. Note this operations to mirror the TFHE-rs implmementation.";
}

def CGGI_ShiftLeftOp : CGGI_Op<"shl", [
    Pure
]> {
  let arguments = (ins LWECiphertextLike:$lhs, AnyI8:$shiftAmount);
  let results = (outs LWECiphertextLike:$output);
  let summary = "Arithmetic shift to left of a ciphertext by an integer. Note this operations to mirror the TFHE-rs implmementation.";
}

// Two options:
// 1. Allow arith.constant and use an encryption op to bring it to the ciphertext space.
// 2. Use a trivial op where the constant is embedded in the ciphertext.
def CGGI_CreateTrivialOp : CGGI_Op<"create_trivial", [Pure]> {
  let arguments = (ins Builtin_IntegerAttr:$value);
  let results = (outs LWECiphertextLike:$output);
}

def CGGI_CastOp : CGGI_Op<"cast", [Pure, SameOperandsAndResultShape]> {
  let summary = "change the plaintext space of a CGGI ciphertext";

  let description = [{
    "cast" operation to change the plaintext size of a CGGI ciphertext.
    Note this operations is not a standard CGGI operation, but an mirror of the cast op implemented in TFHE-rs.

    Examples:
    ```
    `cggi.cast %c0 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding> to !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding1>`
    ```
  }];

  let arguments = (ins LWECiphertextLike:$input);
  let results = (outs LWECiphertextLike:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}

#endif  // LIB_DIALECT_CGGI_IR_CGGIOPS_TD_
