#ifndef LIB_TRANSFORMS_ANNOTATEPREPROCESSING_ANNOTATEPREPROCESSING_TD_
#define LIB_TRANSFORMS_ANNOTATEPREPROCESSING_ANNOTATEPREPROCESSING_TD_

include "mlir/Pass/PassBase.td"

def AnnotatePreprocessing : Pass<"annotate-preprocessing"> {
  let summary = "Annotate operations with downstream encode IDs";
  let description = [{
    Runs PreprocessingAnalysis and annotates operations with identifiers
    of downstream encode ops they feed into.

    This pass is meant to be a testing/debugging tool. Passes that need to
    analyze what part of the IR is server-side preprocessing will run the
    underlying analysis directly.

    (* example filepath=tests/Transforms/annotate_preprocessing/doctest.mlir *)
  }];
}

#endif  // LIB_TRANSFORMS_ANNOTATEPREPROCESSING_ANNOTATEPREPROCESSING_TD_
