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

      The optional `moduliPerLevel` parameter is useful for variants of CKKS
      that require 32-bit word arithmetic, and employ techniques like composite
      rescaling to achieve intermediate widths of rescaling.
    }];
    let parameters = (ins
      AttrParameter<"int", "The log of the degree of the polynomial modulus.">:$logN,
      AttrParameter<"DenseI64ArrayAttr",
        "The array of moduli whose product is the ciphertext modulus, ordered "
        "from level 0 (the base modulus remaining after all rescalings) up to "
        "the maximum level L (the fresh encryption level).">:$Q,
      AttrParameter<"DenseI64ArrayAttr", "The array of moduli in the special modulus, used for key switching.">:$P,
      AttrParameter<"int", "The log of the default scale.">:$logDefaultScale,
      DefaultValuedParameter<"CKKSEncryptionType", "CKKSEncryptionType::pk",
        "The encryption type (defaults to `pk`).">:$encryptionType,
      DefaultValuedParameter<"CKKSEncryptionTechnique", "CKKSEncryptionTechnique::standard",
        "The encryption technique (defaults to `standard`).">:$encryptionTechnique,
      OptionalParameter<"DenseI32ArrayAttr",
        "How many moduli in `Q` belong to each level, ordered from level 0. "
        "E.g., if `moduliPerLevel = {2, 3}`, then the first level uses moduli "
        "`Q[0], Q[1]`, and the second level uses moduli `Q[2], Q[3], Q[4]`. "
        "When omitted, defaults to all 1s. When provided, all entries must be "
        "positive and their sum must equal `Q.size()`.">:$moduliPerLevel
    );
    let genVerifyDecl = 1;
}

#endif  // LIB_DIALECT_CKKS_IR_CKKSATTRIBUTES_TD_
