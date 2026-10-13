#ifndef LIB_DIALECT_CHEDDAR_TRANSFORMS_CONFIGURECRYPTOCONTEXT_TD_
#define LIB_DIALECT_CHEDDAR_TRANSFORMS_CONFIGURECRYPTOCONTEXT_TD_

include "mlir/Pass/PassBase.td"

def ConfigureCryptoContext : Pass<"cheddar-configure-crypto-context"> {
  let summary = "Configure Cheddar crypto context and generate setup helpers";

  let description = [{
    Generates Cheddar setup helpers (`__setup`, `__keygen`, `__configure`)
    from CKKS scheme parameters and required rotation/bootstrap operations.

    (* example filepath=tests/Dialect/Cheddar/Transforms/doctest.mlir *)
  }];

  let options = [
    Option<"entryFunction", "entry-function", "std::string",
           /*default=*/"\"\"", "Entry function name">,
    Option<"defaultEncryptionLevel", "default-encryption-level", "int",
           /*default=*/"-1", "Default encryption level override">,
    Option<"denseHammingWeight", "dense-hamming-weight", "int",
           /*default=*/"0", "Dense hamming weight for secret keys">,
    Option<"sparseHammingWeight", "sparse-hamming-weight", "int",
           /*default=*/"0", "Sparse hamming weight for secret keys">,
    Option<"numCtsLevels", "num-cts-levels", "int",
           /*default=*/"0", "Coeff-to-slot level budget for bootstrapping (defaults to 3 if not set in cheddar.parameter_set)">,
    Option<"numStcLevels", "num-stc-levels", "int",
           /*default=*/"0", "Slot-to-coeff level budget for bootstrapping (defaults to 3 if not set in cheddar.parameter_set)">,
    Option<"prepareRotationKeysAtUseLevels", "prepare-rotation-keys-at-use-levels", "bool",
           /*default=*/"false", "Prepare rotation keys at the maximum level where each key is used">,
  ];

  let dependentDialects = [
    "mlir::heir::cheddar::CheddarDialect",
    "mlir::tensor::TensorDialect",
    "mlir::func::FuncDialect",
  ];
}

#endif  // LIB_DIALECT_CHEDDAR_TRANSFORMS_CONFIGURECRYPTOCONTEXT_TD_
