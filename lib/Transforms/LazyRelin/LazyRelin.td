#ifndef LIB_TRANSFORMS_LAZY_RELIN_TD_
#define LIB_TRANSFORMS_LAZY_RELIN_TD_

include "mlir/Pass/PassBase.td"

def LazyRelin : Pass <"lazy-relin"> {
    let summary = "Delay relin ops";
    let description = [{
        Scan the IR for scopes to insert relin op
    }];

    let dependentDialects = ["mlir::heir::bgv::BGVDialect"];
}

#endif  // LIB_TRANSFORMS_LAZY_RELIN_TD_
