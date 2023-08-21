#ifndef HEIR_INCLUDE_DIALECT_CGGI_IR_CGGIDIALECT_TD_
#define HEIR_INCLUDE_DIALECT_CGGI_IR_CGGIDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/CommonTypeConstraints.td"

def CGGI_Dialect : Dialect {
  let name = "cggi";
  let summary = "A dialect for types and operations in the CGGI cryptosystem";
  let cppNamespace = "::mlir::heir::cggi";
}

// Rather than define custom types, we use type aliases for tensors
// with encodings specified in the `lwe` dialect.

// LWE Ciphertexts are ranked tensors of integers representing the LWE samples
// and the bias.
def LWECiphertext : TypeConstraint<
  And<[
    RankedTensorOf<[AnySignlessInteger]>.predicate,
    // Check the attribute matches the allowed LWE encoding attributes.
    CPred<[{
    ::llvm::isa<::mlir::heir::lwe::BitFieldEncodingAttr>(
      ::llvm::cast<::mlir::RankedTensorType>($_self).getEncoding()
    )
    }]>
  ]>,
  "CGGI LWE ciphertext, aka a ranked tensor with a supported LWE encoding"
>;

#endif  // HEIR_INCLUDE_DIALECT_CGGI_IR_CGGIDIALECT_TD_
