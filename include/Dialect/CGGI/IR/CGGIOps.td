#ifndef HEIR_INCLUDE_DIALECT_CGGI_IR_CGGIOPS_TD_
#define HEIR_INCLUDE_DIALECT_CGGI_IR_CGGIOPS_TD_

include "CGGIDialect.td"

include "mlir/IR/OpBase.td"
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

class CGGI_BinaryGateOp<string mnemonic>
    : CGGI_Op<mnemonic, [Pure, Commutative, SameOperandsAndResultType]> {
  let arguments = (ins LWECiphertext:$lhs, LWECiphertext:$rhs);
  let results = (outs LWECiphertext:$output);
  let assemblyFormat = "operands attr-dict" ;
}

def CGGI_AndOp : CGGI_BinaryGateOp<"and"> { let summary = "Logical AND of two ciphertexts."; }
def CGGI_OrOp  : CGGI_BinaryGateOp<"or">  { let summary = "Logical OR of two ciphertexts."; }
def CGGI_XorOp : CGGI_BinaryGateOp<"xor"> { let summary = "Logical XOR of two ciphertexts."; }

class CGGI_NotOp : CGGI_Op<"not", [Pure, SameOperandsAndResultType, Involution]> {
  let arguments = (ins LWECiphertext:$input);
  let results = (outs LWECiphertext:$output);
  let assemblyFormat = "operands attr-dict" ;
  let summary = "Logical NOT of two ciphertexts";
}

class CGGI_LutOp<string mnemonic> : CGGI_Op<mnemonic, [Pure, Commutative, SameOperandsAndResultType]> {
  let results = (outs LWECiphertext:$output);
  let assemblyFormat = "`(` operands `)` attr-dict `:` type($output)" ;

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

def CGGI_Lut2Op : CGGI_LutOp<"lut2"> {
  let summary = "A lookup table on two inputs.";
  let arguments = (ins LWECiphertext:$b, LWECiphertext:$a, IndexAttr:$lookup_table);
}

def CGGI_Lut3Op : CGGI_LutOp<"lut3"> {
  let summary = "A lookup table on three inputs.";
  let arguments = (ins LWECiphertext:$c, LWECiphertext:$b, LWECiphertext:$a, IndexAttr:$lookup_table);
}

#endif  // HEIR_INCLUDE_DIALECT_CGGI_IR_CGGIOPS_TD_
