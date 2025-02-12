#ifndef LIB_DIALECT_CKKS_IR_CKKSDIALECT_TD_
#define LIB_DIALECT_CKKS_IR_CKKSDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def CKKS_Dialect : Dialect {
  // The namespace of the dialect.
  // This corresponds to the string provided in `getDialectNamespace`.
  let name = "ckks";

  let description = [{
    The CKKS dialect defines the types and operations of the CKKS cryptosystem.
  }];

  let extraClassDeclaration = [{
    constexpr const static ::llvm::StringLiteral
        kSchemeParamAttrName = "ckks.schemeParam";
  }];

  let cppNamespace = "::mlir::heir::ckks";

  let useDefaultAttributePrinterParser = 1;
}

#endif  // LIB_DIALECT_CKKS_IR_CKKSDIALECT_TD_
