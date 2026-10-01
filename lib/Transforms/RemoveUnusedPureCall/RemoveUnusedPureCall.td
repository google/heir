#ifndef LIB_TRANSFORMS_REMOVEUNUSEDPURECALL_REMOVEUNUSEDPURECALL_TD_
#define LIB_TRANSFORMS_REMOVEUNUSEDPURECALL_REMOVEUNUSEDPURECALL_TD_

include "mlir/Pass/PassBase.td"

def RemoveUnusedPureCall : Pass<"remove-unused-pure-call", "ModuleOp"> {
  let summary = "Remove unused calls to pure functions";
  let description = [{
    Remove unused calls to interface helpers whose bodies have no memory
    effects. Calls with unknown effects are retained.

    (* example filepath=tests/Transforms/remove_unused_pure_call/doctest.mlir *)
  }];
  let dependentDialects = ["mlir::func::FuncDialect"];
}

#endif  // LIB_TRANSFORMS_REMOVEUNUSEDPURECALL_REMOVEUNUSEDPURECALL_TD_
