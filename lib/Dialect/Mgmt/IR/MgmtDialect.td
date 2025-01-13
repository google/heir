#ifndef LIB_DIALECT_MGMT_IR_MGMTDIALECT_TD_
#define LIB_DIALECT_MGMT_IR_MGMTDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def Mgmt_Dialect : Dialect {
  let name = "mgmt";
  let description = [{
    The `mgmt` dialect contains scheme-agnostic ciphertext management ops
    (like relinearize and mod reduce), to enable initial high-level compiler
    passes to perform a first pass at parameter selection, while lower-level
    passes may refine them with scheme-specific information.
  }];

  let cppNamespace = "::mlir::heir::mgmt";

  let extraClassDeclaration = [{
    /// Name of the attribute holding MgmtAttr
    constexpr const static ::llvm::StringLiteral
        kArgMgmtAttrName = "mgmt.mgmt";
  }];


  let useDefaultAttributePrinterParser = 1;
}

#endif  // LIB_DIALECT_MGMT_IR_MGMTDIALECT_TD_
