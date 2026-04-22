#ifndef LIB_DIALECT_CHEDDAR_TRANSFORMS_FUSEOPS_TD_
#define LIB_DIALECT_CHEDDAR_TRANSFORMS_FUSEOPS_TD_

include "mlir/Pass/PassBase.td"

def CheddarFuseOps : Pass<"cheddar-fuse-ops"> {
  let summary = "Fuse CHEDDAR ops into compound GPU kernel operations.";

  let description = [{
    This pass fuses sequences of CHEDDAR ops into compound ops that
    map to single fused GPU kernels in the CHEDDAR library:

    - mult + relinearize + rescale -> hmult (with rescale=true)
    - mult + relinearize -> hmult (with rescale=false)
    - hrot + add -> hrot_add (fused rotation + addition)
    - hconj + add -> hconj_add (fused conjugation + addition)

    These fused operations are significantly faster on GPU as they
    avoid intermediate memory allocations and kernel launch overhead.
  }];

  let dependentDialects = [
    "mlir::heir::cheddar::CheddarDialect",
  ];
}

#endif  // LIB_DIALECT_CHEDDAR_TRANSFORMS_FUSEOPS_TD_
