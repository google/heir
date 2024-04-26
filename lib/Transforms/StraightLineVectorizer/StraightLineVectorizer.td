#ifndef LIB_TRANSFORMS_STRAIGHTLINEVECTORIZER_STRAIGHTLINEVECTORIZER_TD_
#define LIB_TRANSFORMS_STRAIGHTLINEVECTORIZER_STRAIGHTLINEVECTORIZER_TD_

include "mlir/Pass/PassBase.td"

def StraightLineVectorizer : Pass<"straight-line-vectorize"> {
  let summary = "A vectorizer for straight line programs.";
  let description = [{
  This pass ignores control flow and only vectorizes straight-line programs
  within a given region.
  }];

  let dependentDialects = [
    "tensor::TensorDialect",
  ];

  let options = [
    Option<"dialect", "dialect", "std::string", "", "Use this to restrict the "
      "dialect whose ops should be vectorized.">,
  ];
}

#endif  // LIB_TRANSFORMS_STRAIGHTLINEVECTORIZER_STRAIGHTLINEVECTORIZER_TD_
