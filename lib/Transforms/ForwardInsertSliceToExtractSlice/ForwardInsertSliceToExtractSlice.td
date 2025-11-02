#ifndef LIB_TRANSFORMS_FORWARDINSERTSLICETOEXTRACTSLICE_FORWARDINSERTSLICETOEXTRACTSLICE_TD_
#define LIB_TRANSFORMS_FORWARDINSERTSLICETOEXTRACTSLICE_FORWARDINSERTSLICETOEXTRACTSLICE_TD_

include "mlir/Pass/PassBase.td"

def ForwardInsertSliceToExtractSlice : Pass<"forward-insert-slice-to-extract-slice"> {
    let summary = "Forward inserts to extracts within a single block";
    let description = [{
        This pass is similar to forward-store-to-load pass where store ops
        are forwarded load ops; here instead tensor.insert_slice ops are forwarded
        to tensor.extract_slice ops.

        Does not support complex control flow within a block, nor ops with
        arbitrary subregions.
    }];
}

#endif // LIB_TRANSFORMS_FORWARDINSERTSLICETOEXTRACTSLICE_FORWARDINSERTSLICETOEXTRACTSLICE_TD_
