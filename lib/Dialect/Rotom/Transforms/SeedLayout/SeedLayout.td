#ifndef LIB_DIALECT_ROTOM_TRANSFORMS_SEEDLAYOUT_SEEDLAYOUT_TD_
#define LIB_DIALECT_ROTOM_TRANSFORMS_SEEDLAYOUT_SEEDLAYOUT_TD_

include "mlir/Pass/PassBase.td"

def SeedLayout
    : Pass<"rotom-seed-layout", "ModuleOp"> {
  let summary = "Seed tensor_ext.layout with rotom.layout attributes";
  let description = [{
    Walks all SSA values in the module, and assigns an initial layout for each
    value.

    (* example filepath=tests/Dialect/Rotom/Transforms/doctest_seed_layout.mlir *)
  }];
  let dependentDialects = [
    "mlir::heir::rotom::RotomDialect",
    "mlir::heir::tensor_ext::TensorExtDialect",
  ];

  let options = [
    Option<"n", "n", "int64_t", /*default=*/"16384", "Ciphertext slot capacity">
  ];
}

#endif  // LIB_DIALECT_ROTOM_TRANSFORMS_SEEDLAYOUT_SEEDLAYOUT_TD_
