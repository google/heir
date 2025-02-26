#ifndef LIB_DIALECT_BGV_IR_BGVDIALECT_TD_
#define LIB_DIALECT_BGV_IR_BGVDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def BGV_Dialect : Dialect {
  // The namespace of the dialect.
  // This corresponds to the string provided in `getDialectNamespace`.
  let name = "bgv";

  let description = [{
    The BGV dialect defines the types and operations of the BGV and B/FV cryptosystem.

    Due to similarity with the BFV scheme, BGV dialect also represents the B/FV scheme.

    The semantics of bgv dialect operations are determined by the `scheme.bgv` or `scheme.bfv`
    annotation at the module level.

    In B/FV mode, bgv.modulus_switch is an no-op.
  }];

  let extraClassDeclaration = [{
    constexpr const static ::llvm::StringLiteral
        kSchemeParamAttrName = "bgv.schemeParam";
  }];

  let cppNamespace = "::mlir::heir::bgv";

  let useDefaultAttributePrinterParser = 1;
}

#endif  // LIB_DIALECT_BGV_IR_BGVDIALECT_TD_
