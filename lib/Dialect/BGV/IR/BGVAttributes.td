#ifndef LIB_DIALECT_BGV_IR_BGVATTRIBUTES_TD_
#define LIB_DIALECT_BGV_IR_BGVATTRIBUTES_TD_

include "BGVDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

class BGV_Attribute<string attrName, string attrMnemonic>
    : AttrDef<BGV_Dialect, attrName> {
    let summary = "Attribute for BGV";
    let description = [{
        This attribute represents the values for BGV.
    }];
    let mnemonic = attrMnemonic;
    let assemblyFormat = "`<` struct(params) `>`";
}

def BGV_SchemeParam
    : BGV_Attribute<"SchemeParam", "scheme_param"> {
    let summary = "BGV Scheme Parameters";
    let description = [{
      This attribute is used for recording the scheme parameters for CKKS.

      The attribute is a struct with the following fields:
        - `int` logN: The log of the degree of the polynomial modulus.
        - `DenseI64ArrayAttr` Q: The array of primes in the ciphertext modulus.
        - `DenseI64ArrayAttr` P: The array of primes in the special modulus, used for key switching.
        - `int64_t` plaintextModulus: The plaintext modulus.
    }];
    let parameters = (ins
      "int":$logN,
      "DenseI64ArrayAttr":$Q,
      "DenseI64ArrayAttr":$P,
      "int64_t":$plaintextModulus
    );
}

#endif  // LIB_DIALECT_BGV_IR_BGVATTRIBUTES_TD_
