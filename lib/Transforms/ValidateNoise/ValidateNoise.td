#ifndef LIB_TRANSFORMS_VALIDATENOISE_VALIDATENOISE_TD_
#define LIB_TRANSFORMS_VALIDATENOISE_VALIDATENOISE_TD_

include "mlir/Pass/PassBase.td"

def ValidateNoise : Pass<"validate-noise"> {
  let summary = "Validate the HE circuit against a given noise model";
  let description = [{
    This pass validates the noise of the HE circuit against a given noise model.

    The pass expects the scheme parameters to be annotated in the IR. Usually
    this is done by the `generate-param-<scheme>` passes.

    For available noise models, see `generate-param-<scheme>` passes.

    The result should be observed using --debug-only=ValidateNoise.

    Example
    ```bash
    # with commandline --debug-only=ValidateNoise
    Noise Bound: 29.27 Budget: 149.73 Total: 179.00 for value: <block argument> of type 'tensor<8xi16>' at index: 0
    Noise Bound: 29.27 Budget: 149.73 Total: 179.00 for value: <block argument> of type 'tensor<8xi16>' at index: 1
    ```
  }];

  let dependentDialects = [
    "mlir::heir::bgv::BGVDialect",
  ];

  let options = [
    Option<"model", "model", "std::string",
           /*default=*/"\"bgv-noise-kpz21\"", "Noise model to validate against.">,
    Option<"annotateNoiseBound", "annotate-noise-bound", "bool",
           /*default=*/"false", "Annotate the noise bound to the IR.">,
  ];
}

#endif  // LIB_TRANSFORMS_VALIDATENOISE_VALIDATENOISE_TD_
