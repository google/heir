#ifndef LIB_TRANSFORMS_ANNOTATESECRETNESS_ANNOTATESECRETNESS_TD_
#define LIB_TRANSFORMS_ANNOTATESECRETNESS_ANNOTATESECRETNESS_TD_

include "mlir/Pass/PassBase.td"

def AnnotateSecretness : Pass<"annotate-secretness"> {
  let summary = "Annotate secret SSA values in the IR";
  let description = [{
  This pass runs the secretness analysis and annotates the IR with the results,
  saving it into each op's attribute dictionary as secret : <analysis results>
  }];
}

#endif  // LIB_TRANSFORMS_ANNOTATESECRETNESS_ANNOTATESECRETNESS_TD_
