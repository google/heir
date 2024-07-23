#ifndef LIB_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLENUMS_TD_
#define LIB_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLENUMS_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinDialect.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/EnumAttr.td"

include "lib/Dialect/CGGI/IR/BooleanGates.td"

def BOOL_GATE_AND: I32EnumAttrCase<"AND", AND_GATE>;
def BOOL_GATE_NAND: I32EnumAttrCase<"NAND", NAND_GATE>;
def BOOL_GATE_OR: I32EnumAttrCase<"OR", OR_GATE>;
def BOOL_GATE_NOR: I32EnumAttrCase<"NOR", NOR_GATE>;
def BOOL_GATE_XOR: I32EnumAttrCase<"XOR", XOR_GATE>;
def BOOL_GATE_XNOR: I32EnumAttrCase<"XNOR", XNOR_GATE>;

// Enum definition is done in CGGIEnums.td
def TfheRustBool_BooleanGateEnumAttr : I32EnumAttr<"TfheRustBoolGateEnum",
    "An enum attribute representing a TFHE-rs boolean gate using u8 int",
    [ BOOL_GATE_AND, BOOL_GATE_NAND, BOOL_GATE_OR,
      BOOL_GATE_NOR, BOOL_GATE_XOR, BOOL_GATE_XNOR
    ]>
{
  let cppNamespace = "::mlir::heir::tfhe_rust_bool";

}


#endif  // LIB_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLENUMS_TD_
