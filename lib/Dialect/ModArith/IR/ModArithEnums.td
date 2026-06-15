#ifndef LIB_DIALECT_MODARITH_IR_MODARITHENUMS_TD_
#define LIB_DIALECT_MODARITH_IR_MODARITHENUMS_TD_

include "lib/Dialect/ModArith/IR/ModArithDialect.td"
include "mlir/IR/EnumAttr.td"

def ModArith_LiftType_STANDARD : I32EnumAttrCase<"STANDARD", 0, "standard">;
def ModArith_LiftType_CENTERED : I32EnumAttrCase<"CENTERED", 1, "centered">;

def ModArith_LiftTypeEnum : I32EnumAttr<"LiftType", "ModArith lift type",
    [ModArith_LiftType_STANDARD, ModArith_LiftType_CENTERED]> {
  let cppNamespace = "::mlir::heir::mod_arith";
}

#endif  // LIB_DIALECT_MODARITH_IR_MODARITHENUMS_TD_
