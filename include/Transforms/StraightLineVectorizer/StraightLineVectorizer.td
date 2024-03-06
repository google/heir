#ifndef INCLUDE_TRANSFORMS_STRAIGHTLINEVECTORIZER_STRAIGHTLINEVECTORIZER_TD_
#define INCLUDE_TRANSFORMS_STRAIGHTLINEVECTORIZER_STRAIGHTLINEVECTORIZER_TD_

include "mlir/Pass/PassBase.td"

def StraightLineVectorizer : Pass<"straight-line-vectorize"> {
  let summary = "A vectorizer for straight line programs.";
  let description = [{
  This pass ignores control flow and only vectorizes straight-line programs
  within a given region.
  }];
  let dependentDialects = [];
}

#endif  // INCLUDE_TRANSFORMS_STRAIGHTLINEVECTORIZER_STRAIGHTLINEVECTORIZER_TD_
