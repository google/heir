#ifndef LIB_TRANSFORMS_BOOLEANVECTORIZER_BOOLEANVECTORIZER_TD_
#define LIB_TRANSFORMS_BOOLEANVECTORIZER_BOOLEANVECTORIZER_TD_

include "mlir/Pass/PassBase.td"

def BooleanVectorizer : Pass<"boolean-vectorize"> {
  let summary = "Group operations into batch vectorizable operations";
  let description = [{
    This pass is used to group operations into a single batched operation.

    The pass supports any operation that implements the
    `BatchVectorizableOpInterface`. The op interface controls what operations
    can be batched together, and how to construct the resulting batched
    operation.

    Pass is based on the straight-line-vectorizer, but is fundamentally different.

    For e.g., the pass is used by the `FPT` tfhe-rs API, where all boolean gates can be
    batched together (regardless of the type of gate). The batched operation
    produces is a `packed_gates` function that takes the boolean gates as a
    string vector and a left and right vector of ciphertexts. Each boolean gates
    specified in `gates` is then applied element wise to the ciphertext vectors.

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
    "scf::SCFDialect"
  ];
}

#endif  // LIB_TRANSFORMS_BOOLEANVECTORIZER_BOOLEANVECTORIZER_TD_
