#ifndef LIB_TRANSFORMS_ANNOTATEMULDEPTH_ANNOTATEMULDEPTH_TD_
#define LIB_TRANSFORMS_ANNOTATEMULDEPTH_ANNOTATEMULDEPTH_TD_

include "mlir/Pass/PassBase.td"

def AnnotateMulDepth : Pass<"annotate-muldepth"> {
  let summary = "Annotate multiplicative depth in the IR";
  let description = [{
    Debugging helper that runs the multiplicative depth analysis and annotates
    the IR with the results.

    (* example filepath=tests/Transforms/annotate_muldepth/doctest.mlir *)
  }];
}

#endif  // LIB_TRANSFORMS_ANNOTATEMULDEPTH_ANNOTATEMULDEPTH_TD_
