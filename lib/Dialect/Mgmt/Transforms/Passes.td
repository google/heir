#ifndef LIB_DIALECT_MGMT_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_MGMT_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def AnnotateMgmt : Pass<"annotate-mgmt"> {
  let summary = "Annotate MgmtAttr for secret SSA values in the IR";
  let description = [{
  This pass runs the secretness/level/dimension analysis and annotates the IR with the results,
  saving it into each op's attribute dictionary as mgmt.mgmt : <analysis results>
  }];
  let dependentDialects = ["mlir::heir::mgmt::MgmtDialect"];
  let options = [
    Option<"baseLevel", "base-level", "int",
           /*default=*/"0", "Level to start counting from (used by B/FV)">,
    Option<"levelBudget", "level-budget", "int",
           /*default=*/"-1",
           "The maximum allowable level (excluding bootstrap levels) before "
           "it is considered invalid. Default to -1 (use target max level or 40)">,
  ];
}

#endif  // LIB_DIALECT_MGMT_TRANSFORMS_PASSES_TD_
