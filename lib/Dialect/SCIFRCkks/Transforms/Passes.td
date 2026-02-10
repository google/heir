#ifndef LIB_ALL_DIALECT_CORNAMI_SCIFRCKKS_ANALYSIS_PASSES_TD_
#define LIB_ALL_DIALECT_CORNAMI_SCIFRCKKS_ANALYSIS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def CKKSEstimator : Pass<"ckks-tigris-estimator"> {
  let summary = "Estimate resources for the CKKS program on Cornami Tigris chip(s)";
  let description = [{
    Estimate resources for the CKKS program on Cornami Tigris chip(s)
    Particularly provide number of operators in program (CKKS) and FC count, latency of
    each operator in CKKS; provide a cumulative count of resources used in the CKKS program
    when they are implemented in Cornami Tigris chip(s).
  }];
  let dependentDialects = ["mlir::heir::ckks::CKKSDialect","mlir::heir::mod_arith::ModArithDialect",
      "mlir::heir::rns::RNSDialect"];
}

def OpenfheEstimator : Pass<"openfhe-tigris-estimator"> {
  let summary = "Estimate resources for the Openfhe program on Cornami Tigris chip(s)";
  let description = [{
    Estimate resources for the Openfhe program on Cornami Tigris chip(s)
    Particularly provide number of operators in program (Openfhe) and FC count, latency of
    each operator in Openfhe; provide a cumulative count of resources used in the Openfhe program
    when they are implemented in Cornami Tigris chip(s).
  }];
  let dependentDialects = ["mlir::heir::openfhe::OpenfheDialect","mlir::heir::mod_arith::ModArithDialect",
        "mlir::heir::rns::RNSDialect"];
}

#endif  // LIB_ALL_DIALECT_CORNAMI_SCIFRCKKS_ANALYSIS_PASSES_TD_
