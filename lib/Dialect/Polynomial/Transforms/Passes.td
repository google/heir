#ifndef LIB_DIALECT_POLYNOMIAL_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_POLYNOMIAL_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def PolyMulToNTT : Pass <"convert-polynomial-mul-to-ntt", "func::FuncOp"> {
  let summary = "Optimally inserts NTT/INTT into a Polynomial IR";
  let description = [{
    This pass converts polynomial multiplication operations to use the Number
    Theoretic Transform (NTT) domain. It uses a demand-based analysis to
    minimize the number of NTT and INTT operations inserted into the graph,
    rather than a local greedy approach.
    (* example filepath=tests/Dialect/Polynomial/Transforms/poly_mul_to_ntt_optimal_placement.mlir *)
  }];
  let dependentDialects = ["mlir::heir::polynomial::PolynomialDialect"];

  let statistics = [
    Statistic<"numNttsInserted", "num-ntts-inserted", "Number of NTT ops inserted">,
    Statistic<"numInttsInserted", "num-intts-inserted", "Number of INTT ops inserted">
  ];
}

def AttachNTTRoots : Pass <"attach-ntt-roots"> {
  let summary = "Attach appropriate roots of unity to NTTs and INTTs";
  let description = [{
    This pass attaches roots of unity to NTTs and INTTs. It should be run after -convert-polynomial-mul-to-ntt.
    It can be run before or after split-rns-ntt, but it is required before lowering to ModArith.
    (* example filepath=tests/Dialect/Polynomial/Transforms/attach_ntt_roots.mlir *)
  }];
  let dependentDialects = ["mlir::heir::polynomial::PolynomialDialect"];
}

def SplitRNSNTT : Pass <"split-rns-ntt"> {
  let summary = "Split RNS NTTs/INTTs into single-modulus NTTs/INTTs";
  let description = [{
    This pass splits NTTs and INTTs on RNS polynomials into NTTs and INTTS
    on single-modulus polynomials.
    This pass is required before lowering to ModArith.
    (* example filepath=tests/Dialect/Polynomial/Transforms/split_rns_ntt.mlir *)
  }];
  let dependentDialects = ["mlir::heir::polynomial::PolynomialDialect"];
}

#endif  // LIB_DIALECT_POLYNOMIAL_TRANSFORMS_PASSES_TD_
