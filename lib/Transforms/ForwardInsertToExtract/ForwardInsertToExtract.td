#ifndef LIB_TRANSFORMS_FORWARDINSERTTOEXTRACT_FORWARDINSERTTOEXTRACT_TD_
#define LIB_TRANSFORMS_FORWARDINSERTTOEXTRACT_FORWARDINSERTTOEXTRACT_TD_

include "mlir/Pass/PassBase.td"

def ForwardInsertToExtract : Pass<"forward-insert-to-extract"> {
    let summary = "Forward inserts to extracts within a single block";
    let description = [{
        This pass is similar to forward-store-to-load pass where store ops
        are forwarded load ops; here instead tensor.insert ops are forwarded
        to tensor.extract ops.

        Does not support complex control flow within a block, nor ops with
        arbitrary subregions.
    }];
    let dependentDialects = [
        "mlir::affine::AffineDialect",
        "mlir::memref::MemRefDialect",
        "mlir::tensor::TensorDialect"
    ];
}

#endif // LIB_TRANSFORMS_FORWARDINSERTTOEXTRACT_FORWARDINSERTTOEXTRACT_TD_
