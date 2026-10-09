#ifndef LIB_DIALECT_CHEDDAR_TRANSFORMS_FUSEOPS_TD_
#define LIB_DIALECT_CHEDDAR_TRANSFORMS_FUSEOPS_TD_

include "mlir/Pass/PassBase.td"

def CheddarFuseOps : Pass<"cheddar-fuse-ops"> {
  let summary = "Fuse Cheddar operations into compound GPU kernels";

  let description = [{
    Fuses Cheddar operation sequences that scale-snu/cheddar implements as a
    single GPU kernel:

    - mult + relinearize + rescale -> hmult (with rescale=true)
    - mult + relinearize -> hmult (with rescale=false)
    - mult + relinearize_rescale -> hmult (with rescale=true)
    - hrot + add -> hrot_add
    - hconj + add -> hconj_add
  }];

  let dependentDialects = [
    "mlir::heir::cheddar::CheddarDialect",
  ];
}

#endif  // LIB_DIALECT_CHEDDAR_TRANSFORMS_FUSEOPS_TD_
