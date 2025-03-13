#ifndef LIB_DIALECT_BGV_IR_BGVENUMS_TD_
#define LIB_DIALECT_BGV_IR_BGVENUMS_TD_

include "mlir/IR/EnumAttr.td"

def BGV_EncryptionTypeEnum : I32EnumAttr<"BGVEncryptionType", "An enum attribute representing an encryption method", [
  I32EnumAttrCase<"pk", 0>,
  I32EnumAttrCase<"sk", 1>
]> {
    let cppNamespace = "::mlir::heir::bgv";
}

def BGV_EncryptionTechniqueEnum : I32EnumAttr<"BGVEncryptionTechnique", "An enum attribute representing an encryption technique", [
  I32EnumAttrCase<"standard", 0>,
  I32EnumAttrCase<"extended", 1>
]> {
    let cppNamespace = "::mlir::heir::bgv";
}

#endif  // LIB_DIALECT_BGV_IR_BGVENUMS_TD_
