#ifndef LIB_TRANSFORMS_ILP_BOOTSTRAP_PLACEMENT_TD_
#define LIB_TRANSFORMS_ILP_BOOTSTRAP_PLACEMENT_TD_

include "mlir/Pass/PassBase.td"

def ILPBootstrapPlacement : Pass <"ilp-bootstrap-placement"> {
    let summary = "Optimize placement of bootstrap ops using ILP";
    let description = [{
        This pass uses an integer linear program to determine the optimal level
        of each term in the MLIR, and thus the placement of bootstrap and
        modreduce operations.

        The pass runs on [ciphertext-semantic](https://heir.dev/docs/design/layout/#data-semantic-and-ciphertext-semantic-tensors)
        IR (secret.generic with arith ops operating on pre-packed tensors). It
        1) Inserts mgmt.modreduce after each level-consuming op (e.g. mul in
           CKKS, where level drops only at multiplications).
        2) Inserts mgmt.bootstrap at the positions chosen by the ILP.
        3) Inserts mgmt.relinearize after each mul. Resulting order is mul ->
           relinearize -> modreduce, with bootstrap after modreduce or after
           the op where the ILP chose.

        Note: The ILP formulation does not account for a freshly encrypted 
        ciphertext starting at a higher level than the bootstrap waterline. 
        This will be implemented as future work.
    }];

    let dependentDialects = ["mlir::heir::mgmt::MgmtDialect"];

  let options = [
    Option<"bootstrapWaterline",
           "bootstrap-waterline",
           "int",
           /*default=*/"3",
           "Bootstrap waterline (max level). Levels are 0..bootstrap-waterline (inclusive); inputs start at bootstrap-waterline.">,
  ];
}

#endif  // LIB_TRANSFORMS_ILP_BOOTSTRAP_PLACEMENT_TD_
