#ifndef LIB_DIALECT_ROTOM_TRANSFORMS_MATERIALIZETENSOREXTLAYOUT_MATERIALIZETENSOREXTLAYOUT_TD_
#define LIB_DIALECT_ROTOM_TRANSFORMS_MATERIALIZETENSOREXTLAYOUT_MATERIALIZETENSOREXTLAYOUT_TD_

include "mlir/Pass/PassBase.td"

def MaterializeTensorExtLayout
    : Pass<"rotom-materialize-tensor-ext-layout", "ModuleOp"> {
  let summary = "Materialize tensor_ext.layout from rotom.layout attributes";
  let description = [{
    Walks all SSA values in the module, and for each value that has an
    associated `rotom.layout` (via the same rules as
    `findAttributeAssociatedWith`), lowers it to the ISL string form used by
    `tensor_ext.layout`, sets `tensor_ext.layout` on that value, and removes the
    `rotom.layout` attribute. The pass fails if any Rotom layout cannot be
    lowered.

    (* example filepath=tests/Dialect/Rotom/Transforms/materialize_row_major.mlir *)
  }];
  let dependentDialects = [
    "mlir::heir::rotom::RotomDialect",
    "mlir::heir::tensor_ext::TensorExtDialect",
  ];
}

#endif  // LIB_DIALECT_ROTOM_TRANSFORMS_MATERIALIZETENSOREXTLAYOUT_MATERIALIZETENSOREXTLAYOUT_TD_
