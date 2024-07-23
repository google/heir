#ifndef LIB_DIALECT_CGGI_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_CGGI_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def SetDefaultParameters : Pass<"cggi-set-default-parameters"> {
  let summary = "Set default parameters for CGGI ops";
  let description = [{
  This pass adds default parameters to all CGGI ops as `cggi_params` named
  attributes, overriding any existing attribute set with that name.

  This pass is primarily for testing purposes, and as a parameter provider
  before a proper parameter selection mechanism is added. This pass should not
  be used in production.

  The specific parameters are hard-coded in
  `lib/Dialect/CGGI/Transforms/SetDefaultParameters.cpp`.
  }];
  let dependentDialects = ["mlir::heir::cggi::CGGIDialect"];
}

def BooleanLineVectorizer : Pass<"cggi-boolean-line-vectorize"> {
  let summary = "Group different logic gates with the packed API";
  let description = [{
    This pass groups independent logic gates into a single call of the packed
    operations.

    Pass is intended for the `FPT` tfhe-rs API, where `packed_gates` function get a
    the boolean gates are passed as a string vector and a left and right vector of ciphertexts.
    Each boolean gates specified in `gates` is then applied element wise.

    ```
    let outputs_ct = fpga_key.packed_gates(&gates, &ref_to_ct_lefts, &ref_to_ct_rights);
    ```
  }];

  let dependentDialects = [
    "tensor::TensorDialect",
  ];

  // let dependentDialects = ["mlir::heir::cggi::CGGIDialect"];
}

#endif  // LIB_DIALECT_CGGI_TRANSFORMS_PASSES_TD_
