#ifndef INCLUDE_TRANSFORMS_VALIDATENOISE_VALIDATENOISE_TD_
#define INCLUDE_TRANSFORMS_VALIDATENOISE_VALIDATENOISE_TD_

include "mlir/Pass/PassBase.td"

def ValidateNoise : Pass<"validate-noise"> {
  let summary = "Validate the FHE noise growth in the IR";
  let description = [{
  This pass applies a noise propagation analysis to the IR and checks that
  the noise does not grow beyond a feasible maximum, based on the paraneters
  chosen for the FHE dialects involved.

  Requires the ops in the IR to implement `NoisePropagationInterface`.
  }];
}

#endif  // INCLUDE_TRANSFORMS_VALIDATENOISE_VALIDATENOISE_TD_
