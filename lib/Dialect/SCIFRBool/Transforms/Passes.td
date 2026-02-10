#ifndef LIB_ALL_DIALECT_CORNAMI_ANALYSIS_PASSES_TD_
#define LIB_ALL_DIALECT_CORNAMI_ANALYSIS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def CGGIEstimator : Pass<"cggi-tigris-estimator"> {
  let summary = "Estimate resources for the CGGI program on Cornami Tigris chip(s)";
  let description = [{
    Estimate resources for the CGGI program on Cornami Tigris chip(s)
    Particularly provide number of operators in program (CGGI) and FC count, latency of
    each operator in CGGI; provide a cumulative count of resources used in the CGGI program
    when they are implemented in Cornami Tigris chip(s).
  }];

  let dependentDialects = ["mlir::heir::cggi::CGGIDialect"];
}
#endif  // LIB_ALL_DIALECT_CORNAMI_ANALYSIS_PASSES_TD_
