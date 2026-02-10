#ifndef LIB_DIALECT_POLYNOMIAL_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_POLYNOMIAL_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

// TODO(#1095): migrate to mod arith type
def PolyMulToNTT : Pass <"convert-polynomial-mul-to-ntt", "func::FuncOp"> {
  let summary = "Optimally inserts NTT/INTT into a Polynomial IR";
  let description = [{
    This pass converts polynomial multiplication operations to use the Number
    Theoretic Transform (NTT) domain. It uses a demand-based analysis to
    minimize the number of NTT and INTT operations inserted into the graph,
    rather than a local greedy approach.
  }];
  let dependentDialects = ["mlir::heir::polynomial::PolynomialDialect"];

  let statistics = [
    Statistic<"numNttsInserted", "num-ntts-inserted", "Number of NTT ops inserted">,
    Statistic<"numInttsInserted", "num-intts-inserted", "Number of INTT ops inserted">
  ];
}

#endif  // LIB_DIALECT_POLYNOMIAL_TRANSFORMS_PASSES_TD_
