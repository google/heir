#ifndef LIB_DIALECT_PREPROCESSING_CONVERSIONS_PREPROCESSINGTOCHEDDAR_PREPROCESSINGTOCHEDDAR_TD_
#define LIB_DIALECT_PREPROCESSING_CONVERSIONS_PREPROCESSINGTOCHEDDAR_PREPROCESSINGTOCHEDDAR_TD_

include "mlir/Pass/PassBase.td"

def PreprocessingToCheddar : Pass<"preprocessing-to-cheddar", "mlir::ModuleOp"> {
  let summary = "Lower Preprocessing ops to memrefs of Cheddar plaintexts";
  let description = [{
    Converts `preprocessing.storage` to a `memref` of the scalar
    `!cheddar.plaintext` element and lowers `empty`, `store`, and `load`.

    The Cheddar dialect represents a single plaintext as a rank-0
    `tensor<!cheddar.plaintext>`. A memref element type cannot be a tensor, so
    the storage memref holds the unwrapped scalar `!cheddar.plaintext`.

    For move-only payloads (`!cheddar.plaintext`, etc.), `store` lowers to
    `bufferization.materialize_in_destination` targeting a rank-0 subview of
    the storage buffer to enable destination passing style without copying,
    and `load` replaces Cheddar DPS input operands with the storage subview
    directly (preserving zero-copy semantics). Non-move-only rank-0 tensors
    bridge to and from the scalar memref storage via `tensor.extract` and
    `tensor.from_elements`.

    This pass must run after `--lwe-to-cheddar`, which converts storage element
    types from `lwe.lwe_plaintext` to `tensor<!cheddar.plaintext>`.
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::heir::cheddar::CheddarDialect",
    "mlir::memref::MemRefDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_DIALECT_PREPROCESSING_CONVERSIONS_PREPROCESSINGTOCHEDDAR_PREPROCESSINGTOCHEDDAR_TD_
