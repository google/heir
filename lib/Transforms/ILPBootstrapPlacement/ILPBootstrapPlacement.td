#ifndef LIB_TRANSFORMS_ILP_BOOTSTRAP_PLACEMENT_TD_
#define LIB_TRANSFORMS_ILP_BOOTSTRAP_PLACEMENT_TD_

include "mlir/Pass/PassBase.td"

def ILPBootstrapPlacement : Pass <"ilp-bootstrap-placement"> {
    let summary = "Optimize placement of bootstrap ops using ILP";
    let description = [{
        This pass uses an integer linear program to determine feasible levels
        and CKKS-style scales for each term in the MLIR, and thus the placement
        of bootstrap and level-reduction operations.

        The pass runs on [ciphertext-semantic](https://heir.dev/docs/design/layout/#data-semantic-and-ciphertext-semantic-tensors)
        IR (secret.generic with arith ops operating on pre-packed tensors). It
        1) Inserts mgmt.modreduce after each level-consuming op (e.g. mul in
           CKKS, where level drops only at multiplications).
        2) Inserts per-use mgmt.level_reduce ops for edge rescale decisions
           chosen by the ILP.
        3) Inserts mgmt.bootstrap at the positions chosen by the ILP.
        4) Inserts mgmt.relinearize after each mul. Resulting order is mul ->
           relinearize -> modreduce, with bootstrap after modreduce or after
           the op where the ILP chose.

        The ILP formulation is inspired by [Orbit](https://eprint.iacr.org/2026/213.pdf).

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
    Option<"scaleWaterline",
           "scale-waterline",
           "int",
           /*default=*/"40",
           "Minimum CKKS scale budget used by the Orbit-inspired scale constraints.">,
    Option<"scaleFactorBits",
           "scale-factor-bits",
           "int",
           /*default=*/"51",
           "Scale bits dropped by one rescale/modreduce in the Orbit-inspired scale constraints.">,
    Option<"bootstrapLevelLowerBound",
           "bootstrap-level-lower-bound",
           "int",
           /*default=*/"0",
           "Minimum input level at which bootstrap is allowed in the Orbit-inspired scale constraints.">,
    Option<"orbitCostModel",
           "orbit-cost-model",
           "std::string",
           /*default=*/"""",
           "Path to an Orbit JSON cost model. When provided, bootstrap-cost and rescale-cost are loaded from latencyTable.">,
    Option<"bootstrapCost",
           "bootstrap-cost",
           "int",
           /*default=*/"69320650",
           "Cost of one bootstrap in the ILP objective. Default is the positive-latency average from Orbit's profiled 64k Lattigo base cost model.">,
    Option<"rescaleCost",
           "rescale-cost",
           "int",
           /*default=*/"40988",
           "Cost of one level-reduction/rescale in the ILP objective. Default is the positive-latency average from Orbit's profiled 64k Lattigo base cost model.">,
    Option<"useOrbitCompression",
           "use-orbit-compression",
           "bool",
           /*default=*/"true",
           "When true, add Orbit-inspired structural compression constraints so equivalent ops share ILP state and bootstrap decisions.">,
  ];
}

#endif  // LIB_TRANSFORMS_ILP_BOOTSTRAP_PLACEMENT_TD_
