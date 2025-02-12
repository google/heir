#ifndef LIB_TRANSFORMS_VALIDATENOISE_VALIDATENOISE_TD_
#define LIB_TRANSFORMS_VALIDATENOISE_VALIDATENOISE_TD_

include "mlir/Pass/PassBase.td"

def ValidateNoise : Pass<"validate-noise"> {
  let summary = "Validate the HE circuit against a given noise model";
  let description = [{
    This pass validates the noise of the HE circuit against a given noise model.

    Currently the pass works for BGV scheme, and there are two noise models
    available: "bgv-noise-by-bound-coeff-average-case{-pk,-sk}",
    "bgv-noise-by-bound-coeff-worst-case{-pk,-sk}" and
    "bgv-noise-by-variance-coeff{-pk,-sk}".

    The first two models are taken from KPZ21, and they work by bounding
    the coefficient embedding of the ciphertexts. The difference
    of the two models is expansion factor used for multiplication
    of the coefficients, the first being `2\sqrt{N}` and the second
    being `N`. The `-pk`/`-sk` suffixes assume the input ciphertexts are
    encrypted using the public/secret key.

    The third model is taken from MP24. It works by tracking the variance
    of the coefficient embedding of the ciphertexts. This gives a more accurate
    noise estimate, but it may give underestimates in some cases. See the paper
    for more details.

    This pass is experimental. The result should be observed using
    --debug-only=ValidateNoise.

    This pass relies on the presence of the `mgmt` dialect ops to model
    relinearize/modreduce, and it relies on `mgmt.mgmt` attribute to determine
    the ciphertext level/dimension. These ops and attributes can be added by
    a pass like `--secret-insert-mgmt-<scheme>` and `--annotate-mgmt`.

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
           /*default=*/"", "Noise model to validate against.">,
    Option<"plaintextModulus", "plaintext-modulus", "int64_t",
           /*default=*/"65537", "Plaintext modulus.">,
  ];
}

#endif  // LIB_TRANSFORMS_VALIDATENOISE_VALIDATENOISE_TD_
