#ifndef LIB_TRANSFORMS_SCRATCHPADPASS_SCRATCHPADPASS_TD_
#define LIB_TRANSFORMS_SCRATCHPADPASS_SCRATCHPADPASS_TD_

include "mlir/Pass/PassBase.td"

def ScratchpadPass : Pass<"scratchpad"> {
  let summary = "An empty pass shell for experimentation purposes.";
  let description = [{
  }];
  let dependentDialects = [
    // For any ops that this pass creates, the dialect for that op
    // should be listed here as a dependent dialect, and then the
    // include for that dialect must be added to ScratchpadPass.h
    // before the GEN_PASS_DECL include.
  ];
}

#endif  // LIB_TRANSFORMS_SCRATCHPADPASS_SCRATCHPADPASS_TD_
