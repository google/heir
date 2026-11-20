#ifndef LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPU_ENUMS_TD_
#define LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPU_ENUMS_TD_

include "lib/Dialect/CheddarGPU/IR/CheddarGPUDialect.td"
include "mlir/IR/EnumAttr.td"

// === Form Enum ===

def CheddarGPU_Form_Coeff : I32EnumCase<"Coeff", 0, "coeff">;
def CheddarGPU_Form_CoeffCentered : I32EnumCase<"CoeffCentered", 1, "coeff_centered">;
def CheddarGPU_Form_Eval : I32EnumCase<"Eval", 2, "eval">;

def CheddarGPU_FormEnum : I32Enum<"Form", "Cheddar polynomial representation form", [
  CheddarGPU_Form_Coeff,
  CheddarGPU_Form_CoeffCentered,
  CheddarGPU_Form_Eval
]> {
  let cppNamespace = "::mlir::heir::cheddar_gpu";
}

// === INTT Epilogue Enum ===

def CheddarGPU_INTTEpilogue_None : I32EnumCase<"None", 0, "none">;
def CheddarGPU_INTTEpilogue_MultConst : I32EnumCase<"MultConst", 1, "mult_const">;
def CheddarGPU_INTTEpilogue_MultConstNormalize : I32EnumCase<"MultConstNormalize", 2, "mult_const_normalize">;

def CheddarGPU_INTTEpilogueEnum : I32Enum<"INTTEpilogue", "INTT epilogue post-processing mode", [
  CheddarGPU_INTTEpilogue_None,
  CheddarGPU_INTTEpilogue_MultConst,
  CheddarGPU_INTTEpilogue_MultConstNormalize
]> {
  let cppNamespace = "::mlir::heir::cheddar_gpu";
}

// === Hoist Phase Kind Enum ===

def CheddarGPU_HoistPhaseKind_CTS : I32EnumAttrCase<"CTS", 0, "cts">;
def CheddarGPU_HoistPhaseKind_STC : I32EnumAttrCase<"STC", 1, "stc">;

def CheddarGPU_HoistPhaseKindEnum : I32EnumAttr<"HoistPhaseKind", "Hoist phase kind (cts or stc)", [
  CheddarGPU_HoistPhaseKind_CTS,
  CheddarGPU_HoistPhaseKind_STC
]> {
  let cppNamespace = "::mlir::heir::cheddar_gpu";
  let genSpecializedAttr = 0;
}

#endif  // LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPU_ENUMS_TD_
