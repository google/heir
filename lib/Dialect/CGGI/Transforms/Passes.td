#ifndef LIB_DIALECT_CGGI_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_CGGI_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def DecomposeOperations : Pass<"cggi-decompose-operations"> {
  let summary = "Expands CGGI operations into LWE operations and programmable bootstraps";
  let description = [{
    This pass expands high level CGGI operations (e.g. LUT2, XOR, etc.).

    If the option `expand-lincomb` is set, the expansion will continue into the
    component LWE scalar operations and a programmable bootstrap operation.
    Otherwise, the expansion will be stop at the `cggi.lut_lincomb` level. By
    default, `expand-lincomb` is true.

    For example, a LUT3 operation is composed of three LWE ciphertext inputs $c,
    b, a$ (in MSB to LSB ordering) which must be combined via the linear
    combination $4 * c + 2 * b + a$ before being fed into a programmable
    bootstrap defined by the lookup table.

    This pass supports XOR, LUT2, LUT3, and LutLincomb operations.
  }];
  let dependentDialects = ["mlir::heir::cggi::CGGIDialect"];

  let options = [
    Option<"expandLincomb", "expand-lincomb", "bool",
           /*default=*/"true", "Expand lincomb operations to the PBS and scalar level">
  ];
}

def BooleanVectorizer : Pass<"cggi-boolean-vectorize"> {
  let summary = "Group different logic gates with the packed API";
  let description = [{
    This pass groups independent logic gates into a single call of the packed
    operations. Pass is based on the straight-line-vectorizer, but is fundamentally different.
    This pass combines any type of boolean gates and is not restricted to combining the same type of gate operand.

    Pass is intended for the `FPT` tfhe-rs API, where `packed_gates` function get a
    the boolean gates are passed as a string vector and a left and right vector of ciphertexts.
    Each boolean gates specified in `gates` is then applied element wise.

    ```
    let outputs_ct = fpga_key.packed_gates(&gates, &ref_to_ct_lefts, &ref_to_ct_rights);
    ```
  }];

  let options = [
    Option<"parallelism", "parallelism", "int",
           /*default=*/"0", "Parallelism factor for batching. 0 is infinite parallelism">
  ];

  let dependentDialects = [
    "tensor::TensorDialect",
  ];
}

#endif  // LIB_DIALECT_CGGI_TRANSFORMS_PASSES_TD_
