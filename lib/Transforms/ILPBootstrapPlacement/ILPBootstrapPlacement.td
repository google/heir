#ifndef LIB_TRANSFORMS_ILP_BOOTSTRAP_PLACEMENT_TD_
#define LIB_TRANSFORMS_ILP_BOOTSTRAP_PLACEMENT_TD_

include "mlir/Pass/PassBase.td"

def ILPBootstrapPlacement : Pass <"ilp-bootstrap-placement"> {
    let summary = "Optimize placement of bootstrap ops using ILP";
    let description = [{
        This pass uses an integer linear program to determine the optimal level
        of each term in the MLIR, and thus the placement of bootstrap and
        modreduce operations. It does not have any dependencies on the existing
        secret-insert-mgmt-ckks pass.

        The pass runs on raw IR (secret.generic with arith ops etc.). It
        solves for the level of every value. Then:
        1) Inserts mgmt.modreduce after each level-consuming op (e.g. mul in
           CKKS, where level drops only at multiplications).
        2) Inserts mgmt.bootstrap at the positions chosen by the ILP.
        3) Inserts mgmt.relinearize after each mul. Resulting order is mul ->
           relinearize -> modreduce, with bootstrap after modreduce or after
           the op where the ILP chose.
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
