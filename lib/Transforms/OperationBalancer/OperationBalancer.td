#ifndef LIB_TRANSFORMS_OPERATIONBALANCER_OPERATIONBALANCER_TD_
#define LIB_TRANSFORMS_OPERATIONBALANCER_OPERATIONBALANCER_TD_

include "mlir/Pass/PassBase.td"

def OperationBalancer : Pass<"operation-balancer"> {
  let summary = "This pass balances addition and multiplication operations.";

  let dependentDialects = [
    "arith::ArithDialect"
  ];

  let description = [{
  This pass examines a tree or graph of add and multiplication operations and
  balances them to minimize the depth of the tree. This exposes better parallelization
  and reducing the multiplication depth can decrease the parameters used in FHE,
  which improves performance. This pass is not necessarily optimal, as there may
  be intermediate computations that this pass does not optimally minimize the depth for.

  The algorithm is to analyze a graph of addition operations and do a depth-first
  search for the operands (from the last computed values in the graph). If there
  are intermediate computations that are used more than once, then the pass
  treats that computation as its own tree to balance instead of trying to minimize
  the global depth of the tree.

  This pass only runs on addition and multiplication operations on the arithmetic
  dialect that are encapsulated inside a secret.generic.

  This pass was inspired by section 2.6 of ['EVA Improved: Compiler and Extension
  Library for CKKS' by Chowdhary et al](https://eprint.iacr.org/2021/1505).
  }];
}

#endif  // LIB_TRANSFORMS_OPERATIONBALANCER_OPERATIONBALANCER_TD_
