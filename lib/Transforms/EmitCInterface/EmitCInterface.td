#ifndef LIB_TRANSFORMS_EMITCINTERFACE_EMITCINTERFACE_TD_
#define LIB_TRANSFORMS_EMITCINTERFACE_EMITCINTERFACE_TD_

include "mlir/Pass/PassBase.td"

def EmitCInterface : Pass<"emit-c-interface"> {
  let summary = "Adds llvm.emit_c_interface to each external function.";
}

#endif  // LIB_TRANSFORMS_EMITCINTERFACE_EMITCINTERFACE_TD_
