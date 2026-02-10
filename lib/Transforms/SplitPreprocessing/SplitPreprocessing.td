#ifndef LIB_TRANSFORMS_SPLITPREPROCESSING_SPLITPREPROCESSING_TD_
#define LIB_TRANSFORMS_SPLITPREPROCESSING_SPLITPREPROCESSING_TD_

include "mlir/Pass/PassBase.td"

def SplitPreprocessing : Pass<"split-preprocessing", "mlir::ModuleOp"> {
  let summary = "Splits a function into a preprocessing and a main part";
  let description = [{
    This pass splits a function into a preprocessing and a main part. The
    preprocessing part is executed before the main workload function (the
    preprocessed function) and its results are passed to the preprocessed
    function. This is used to allow packing of plaintexts to be done in advance
    of the main workload, and passed in as arguments.

    The pass identifies all RLWE encoded cleartexts as plaintexts and moves all
    the operations to produce those plaintexts into the preprocessing function.
    The preprocessing function thus returns all the plaintexts created in the
    program. The preprocessing function takes any cleartexts used to create
    plaintexts as inputs.

    (* example filepath=tests/Transforms/split_preprocessing/split_preprocessing.mlir *)
  }];
}

#endif  // LIB_TRANSFORMS_SPLITPREPROCESSING_SPLITPREPROCESSING_TD_
