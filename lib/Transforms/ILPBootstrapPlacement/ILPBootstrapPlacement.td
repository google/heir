#ifndef LIB_TRANSFORMS_ILP_BOOTSTRAP_PLACEMENT_TD_
#define LIB_TRANSFORMS_ILP_BOOTSTRAP_PLACEMENT_TD_

include "mlir/Pass/PassBase.td"

def ILPBootstrapPlacement : Pass <"ilp-bootstrap-placement"> {
    let summary = "Optimize placement of bootstrap ops using ILP";
    let description = [{
        This pass uses an integer linear program to determine feasible levels
        and Orbit-style scales for each term in the MLIR, and thus the
        placement of mod-down, rescale, upscale, and bootstrap operations.

        The pass runs on
        [ciphertext-semantic](https://heir.dev/docs/design/layout/#data-semantic-and-ciphertext-semantic-tensors)
        IR (`secret.generic` with `arith` ops operating on pre-packed
        tensors). It does the following:

        1. Inserts `mgmt.level_reduce` for decoded level-only mod-down.
        2. Inserts `mgmt.adjust_scale` for decoded upscale.
        3. Inserts `mgmt.modreduce` for decoded rescale. In CKKS lowering,
           `mgmt.modreduce` becomes `ckks.rescale`.
        4. Inserts `mgmt.bootstrap` at the positions chosen by the ILP.
        5. Inserts `mgmt.relinearize` after each mul. Resulting order is mul ->
           relinearize, with rescale/bootstrap chains inserted after
           relinearize when the ILP chooses them on a mul result.

        The ILP formulation is inspired by [Orbit](https://eprint.iacr.org/2026/213.pdf).
        CKKS scale modeling is selected from CKKS module attributes. Modules
        without CKKS attributes use the generic level-only model, where the
        effective waterline scale and rescaling factor are equal.
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
           "Orbit waterline/base scale bits (Sw). Non-CKKS level-only mode uses this for both Sw and Sf.">,
    Option<"scaleFactorBits",
           "scale-factor-bits",
           "int",
           /*default=*/"51",
           "Orbit rescale factor bits (Sf), i.e. scale bits dropped by one rescale/modreduce.">,
    Option<"bootstrapLevelLowerBound",
           "bootstrap-level-lower-bound",
           "int",
           /*default=*/"0",
           "Minimum input level at which bootstrap is allowed in the Orbit-inspired scale constraints.">,
    Option<"orbitCostModel",
           "orbit-cost-model",
           "std::string",
           /*default=*/"""",
           "Path to a JSON cost model. When provided, bootstrap-cost and rescale-cost are loaded from latencyTable.bootstrap and latencyTable.rescale. See lib/Transforms/ILPBootstrapPlacement/README.md for the schema.">,
    Option<"bootstrapCost",
           "bootstrap-cost",
           "int",
           /*default=*/"69320650",
           "Cost of one bootstrap in the ILP objective. Default is the positive-latency average from Orbit's profiled 64k Lattigo base cost model.">,
    Option<"rescaleCost",
           "rescale-cost",
           "int",
           /*default=*/"40988",
           "Cost of one ILP-chosen rescale in the objective. This is loaded from Orbit's rescale latency when orbit-cost-model is provided. Default is the positive-latency average from Orbit's profiled 64k Lattigo base cost model.">,
  ];
}

#endif  // LIB_TRANSFORMS_ILP_BOOTSTRAP_PLACEMENT_TD_
