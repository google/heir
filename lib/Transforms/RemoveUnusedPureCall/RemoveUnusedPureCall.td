#ifndef LIB_TRANSFORMS_REMOVEUNUSEDPURECALL_REMOVEUNUSEDPURECALL_TD_
#define LIB_TRANSFORMS_REMOVEUNUSEDPURECALL_REMOVEUNUSEDPURECALL_TD_

include "mlir/Pass/PassBase.td"

def RemoveUnusedPureCall : Pass<"remove-unused-pure-call", "ModuleOp"> {
  let summary = "Remove unused calls to pure functions";
  let description = [{
    This pass removes calls to functions that are considered "pure" if the
    results of the call are unused.

    A function is considered pure if it has an attribute with the `client.`
    namespace.

    (* example filepath=tests/Transforms/remove_unused_pure_call/doctest.mlir *)
  }];
  let dependentDialects = ["mlir::func::FuncDialect"];
}

#endif  // LIB_TRANSFORMS_REMOVEUNUSEDPURECALL_REMOVEUNUSEDPURECALL_TD_
