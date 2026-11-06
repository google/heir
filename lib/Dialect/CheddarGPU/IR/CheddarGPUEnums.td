#ifndef LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPU_ENUMS_TD_
#define LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPU_ENUMS_TD_

include "lib/Dialect/CheddarGPU/IR/CheddarGPUDialect.td"
include "mlir/IR/EnumAttr.td"

// === Form Enum ===

def CheddarGPU_Form_Coeff : I32EnumAttrCase<"Coeff", 0, "coeff">;
def CheddarGPU_Form_CoeffCentered : I32EnumAttrCase<"CoeffCentered", 1, "coeff_centered">;
def CheddarGPU_Form_Eval : I32EnumAttrCase<"Eval", 2, "eval">;

def CheddarGPU_FormEnum : I32EnumAttr<"Form", "Cheddar polynomial representation form", [
  CheddarGPU_Form_Coeff,
  CheddarGPU_Form_CoeffCentered,
  CheddarGPU_Form_Eval
]> {
  let cppNamespace = "::mlir::heir::cheddar_gpu";
  let genSpecializedAttr = 0;
}

// === INTT Epilogue Enum ===

def CheddarGPU_INTTEpilogue_None : I32EnumAttrCase<"None", 0, "none">;
def CheddarGPU_INTTEpilogue_MultConst : I32EnumAttrCase<"MultConst", 1, "mult_const">;
def CheddarGPU_INTTEpilogue_MultConstNormalize : I32EnumAttrCase<"MultConstNormalize", 2, "mult_const_normalize">;

def CheddarGPU_INTTEpilogueEnum : I32EnumAttr<"INTTEpilogue", "INTT epilogue post-processing mode", [
  CheddarGPU_INTTEpilogue_None,
  CheddarGPU_INTTEpilogue_MultConst,
  CheddarGPU_INTTEpilogue_MultConstNormalize
]> {
  let cppNamespace = "::mlir::heir::cheddar_gpu";
  let genSpecializedAttr = 0;
}

#endif  // LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPU_ENUMS_TD_
