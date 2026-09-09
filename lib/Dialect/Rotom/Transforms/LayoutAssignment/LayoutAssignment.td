#ifndef LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_LAYOUTASSIGNMENT_TD_
#define LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_LAYOUTASSIGNMENT_TD_

include "mlir/Pass/PassBase.td"

def LayoutAssignment
    : Pass<"rotom-assign-layout", "ModuleOp"> {
  let summary = "Assign concrete Rotom layouts from seeded candidate layouts";
  let description = [{
    Consumes `rotom.seed` candidate layout attributes and assigns one concrete
    `rotom.layout` to each participating SSA value. The pass performs a
    deterministic candidate propagation over the SSA graph and records selected
    layouts using the same value-associated attribute convention consumed by
    `rotom-materialize-tensor-ext-layout`.
  }];
  let dependentDialects = [
    "mlir::heir::rotom::RotomDialect",
  ];
}

#endif  // LIB_DIALECT_ROTOM_TRANSFORMS_LAYOUTASSIGNMENT_LAYOUTASSIGNMENT_TD_
