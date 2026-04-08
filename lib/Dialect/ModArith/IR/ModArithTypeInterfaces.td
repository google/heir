#ifndef LIB_DIALECT_MODARITH_IR_MODARITHTYPEINTERFACES_TD_
#define LIB_DIALECT_MODARITH_IR_MODARITHTYPEINTERFACES_TD_

include "mlir/IR/BuiltinTypeInterfaces.td"

def ModQTypeInterface : TypeInterface<"ModQTypeInterface"> {
  let cppNamespace = "::mlir::heir::mod_arith";
  let description = [{
    Interface for types that support modular arithmetic semantics.
  }];
  let methods = [
    InterfaceMethod<
        "Returns the lowered storage type for this modular type. Plain ModArith types lower to an integer type; RNS types lower to a rank-1 tensor whose extent is the residue count and whose element type is the residue storage type.",
        "::mlir::Type", "getLoweringType", (ins)>,
    InterfaceMethod<
        "Returns the residue type at the given index.",
        "::mlir::Type", "getResidueType", (ins "unsigned":$index)>
  ];
}

#endif  // LIB_DIALECT_MODARITH_IR_MODARITHTYPEINTERFACES_TD_
