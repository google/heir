#ifndef LIB_DIALECT_JAXITEWORD_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_JAXITEWORD_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def JaxiteCkksParameterSelection : Pass<"jaxite-ckks-parameter-selection"> {
  // FIXME: add add summary/description
  let summary = "";
  let description = [{
  }];
  let dependentDialects = ["mlir::heir::jaxite_word::JaxiteWordDialect"];
}

#endif  // LIB_DIALECT_JAXITEWORD_TRANSFORMS_PASSES_TD_
