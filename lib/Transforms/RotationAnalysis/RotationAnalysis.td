#ifndef LIB_TRANSFORMS_ROTATIONANALYSIS_ROTATIONANALYSIS_TD_
#define LIB_TRANSFORMS_ROTATIONANALYSIS_ROTATIONANALYSIS_TD_

include "mlir/Pass/PassBase.td"

def RotationAnalysisPass : Pass<"rotation-analysis", "ModuleOp"> {
  let summary = "Run RotationAnalysis and annotate the module with discovered indices.";
  let description = [{
    This pass runs RotationAnalysis and adds an attribute `rotation_indices` to
    the module containing all discovered rotation indices.
  }];
}

#endif  // LIB_TRANSFORMS_ROTATIONANALYSIS_ROTATIONANALYSIS_TD_
