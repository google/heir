#ifndef LIB_ALL_DIALECT_CORNAMI_TRANSFORMS_SCIFRBOOLEMITTER_TD_
#define LIB_ALL_DIALECT_CORNAMI_TRANSFORMS_SCIFRBOOLEMITTER_TD_

include "mlir/Pass/PassBase.td"

def CGGIEmitter : Pass<"emit-cggi-to-scifrbool"> {
  let summary = "Emit SCIFRBool C++ code from cggi";

  let dependentDialects = ["mlir::heir::cggi::CGGIDialect"];
}
#endif  // LIB_ALL_DIALECT_CORNAMI_TRANSFORMS_SCIFRBOOLEMITTER_TD_
