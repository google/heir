#ifndef LIB_DIALECT_CGGI_IR_CGGIOPS_TD_
#define LIB_DIALECT_CGGI_IR_CGGIOPS_TD_

include "lib/Dialect/HEIRInterfaces.td"
include "lib/Dialect/CGGI/IR/CGGIDialect.td"
include "lib/Dialect/CGGI/IR/CGGIAttributes.td"

include "lib/Dialect/Polynomial/IR/PolynomialAttributes.td"
include "lib/Dialect/LWE/IR/LWETypes.td"

include "mlir/IR/OpBase.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
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

include "lib/Dialect/CGGI/IR/CGGIBinOps.td"
include "lib/Dialect/CGGI/IR/CGGIPBSOps.td"

/**************************
 * MAINTENANCE OPERATIONS *
 **************************/

// Two options:
// 1. Allow arith.constant and use an encryption op to bring it to the ciphertext space.
// 2. Use a trivial op where the constant is embedded in the ciphertext.
def CGGI_CreateTrivialOp : CGGI_Op<"create_trivial", [Pure]> {
  let arguments = (ins Builtin_IntegerAttr:$value);
  let results = (outs NewLWECiphertextLike:$output);
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

  let arguments = (ins NewLWECiphertextLike:$input);
  let results = (outs NewLWECiphertextLike:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}

#endif  // LIB_DIALECT_CGGI_IR_CGGIOPS_TD_
