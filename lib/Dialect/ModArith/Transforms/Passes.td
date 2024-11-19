#ifndef LIB_DIALECT_MODARITH_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_MODARITH_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def ConvertToMac : Pass<"mod-arith-to-mac"> {
  let summary = "Finds consecutive ModArith mul and add operations and converts them to a Mac operation";
  let description = [{
  Walks over the programs to find Add operations, it checks if the LHS operands originates from a mul operation. If so, it converts the Add operation to a Mac operation and removes the mul operation.
  }];
  let dependentDialects = ["mlir::heir::mod_arith::ModArithDialect"];
}

#endif  // LIB_DIALECT_MODARITH_TRANSFORMS_PASSES_TD_
