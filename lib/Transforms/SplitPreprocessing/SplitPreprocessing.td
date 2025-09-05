#ifndef LIB_TRANSFORMS_SPLITPREPROCESSING_SPLITPREPROCESSING_TD_
#define LIB_TRANSFORMS_SPLITPREPROCESSING_SPLITPREPROCESSING_TD_

include "mlir/Pass/PassBase.td"

def SplitPreprocessing : Pass<"split-preprocessing", "mlir::func::FuncOp"> {
  let summary = "Splits a function into a preprocessing and a main part";
  let description = [{
    This pass splits a function into a preprocessing and a main part. The
    preprocessing part is executed before the main function and its results are
    passed to the main function. This is used to allow packing of
    plaintexts to be done in advance of the main workload, and passed in as
    arguments. Only plaintexts of `arith.constant` ops are currently extracted
    for preprocessing.

    (* example filepath=tests/Transforms/split_preprocessing/split_preprocessing.mlir *)
  }];
}

#endif  // LIB_TRANSFORMS_SPLITPREPROCESSING_SPLITPREPROCESSING_TD_
