#ifndef LIB_DIALECT_CKKS_IR_CKKSATTRIBUTES_TD_
#define LIB_DIALECT_CKKS_IR_CKKSATTRIBUTES_TD_

include "CKKSDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

class CKKS_Attribute<string attrName, string attrMnemonic>
    : AttrDef<CKKS_Dialect, attrName> {
    let mnemonic = attrMnemonic;
    let assemblyFormat = "`<` struct(params) `>`";
}

def CKKS_SchemeParam
    : CKKS_Attribute<"SchemeParam", "scheme_param"> {
    let summary = "CKKS Scheme Parameters";
    let description = [{
      This attribute is used for recording the scheme parameters for CKKS.

      The attribute is a struct with the following fields:
        - `int` logN: The log of the degree of the polynomial modulus.
        - `DenseI64ArrayAttr` Q: The array of primes in the ciphertext modulus.
        - `DenseI64ArrayAttr` P: The array of primes in the special modulus, used for key switching.
        - `int` logDefaultScale: The log of the default scale.
    }];
    let parameters = (ins
      "int":$logN,
      "DenseI64ArrayAttr":$Q,
      "DenseI64ArrayAttr":$P,
      "int":$logDefaultScale
    );
}

#endif  // LIB_DIALECT_CKKS_IR_CKKSATTRIBUTES_TD_
