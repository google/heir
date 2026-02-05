#ifndef LIB_TRANSFORMS_CONVERTTOCIPHERTEXTSEMANTICS_CONVERTTOCIPHERTEXTSEMANTICS_TD_
#define LIB_TRANSFORMS_CONVERTTOCIPHERTEXTSEMANTICS_CONVERTTOCIPHERTEXTSEMANTICS_TD_

include "mlir/Pass/PassBase.td"

def ConvertToCiphertextSemantics : Pass<"convert-to-ciphertext-semantics"> {
  let summary = "Converts programs with tensor semantics to ciphertext semantics";
  let description = [{
  This pass performs two inherently intertwined transformations:

  1. Convert a program from tensor semantics to ciphertext semantics, explained below.
  2. Implement ops defined on tensor-semantic types in terms of ops defined on
     ciphertext-semantic types.

  A program is defined to have _tensor semantics_ if the tensor-typed values
  are manipulated according to standard MLIR tensor operations and semantics.

  A program is defined to have _ciphertext semantics_ if the tensor-typed
  values correspond to tensors of FHE ciphertexts, where the last dimension of
  the tensor type is the number of ciphertext slots.

  For example, a tensor of type `tensor<32x32xi16>` with tensor semantics might
  be converted by this pass, depending on the pass options, to a single
  ciphertext-semantics `tensor<65536xi16>`. A larger tensor might, depending on
  the layout chosen by earlier passes, be converted to a `tensor<4x32768xi16>`,
  where the trailing dimension corresponds to the number of slots in the
  ciphertext.

  Tensors with ciphertext semantics can be thought of as an intermediate step
  between lowering from tensor types with tensor semantics to concrete `lwe`
  dialect ciphertext types in a particular FHE scheme. Having this intermediate
  step is useful because some optimizations are easier to implement, and can be
  implemented more generically, in the abstract FHE computational model
  where the data types are large tensors, and the operations are SIMD additions,
  multiplications, and cyclic rotations.

  Function arguments and return values are annotated with the original tensor
  type in the `secret.original_type` attribute. This enables later lowerings
  to implement appropriate encoding and decoding routines for FHE schemes.

  The second role of this pass is to implement FHE kernels for various high-level
  tensor operations, such as `linalg.matvec`. This must happen at the same time
  as the type conversion because the high-level ops like `linalg.matvec` are
  not well-defined on ciphertext-semantic tensors, while their implementation
  as SIMD/rotation ops are not well-defined on tensor-semantic tensors.

  (* example filepath=tests/Transforms/convert_to_ciphertext_semantics/doctest.mlir *)
  }];
  let dependentDialects = [
    "mlir::heir::tensor_ext::TensorExtDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
    "mlir::scf::SCFDialect"
  ];

  let options = [
    // TODO(#4102): reevaluate flag name
    Option<
      "ciphertextSize",
      "ciphertext-size",
      "int",
      /*default=*/"1024",
      "Power of two length of the ciphertexts the data is packed in."
    >,
    Option<
      "experimentalDisableLoopUnroll",
      "experimental-disable-loop-unroll",
      "bool",
      /*default=*/"false",
      "Keep the kernels rolled."
    >,
  ];
}

#endif  // LIB_TRANSFORMS_CONVERTTOCIPHERTEXTSEMANTICS_CONVERTTOCIPHERTEXTSEMANTICS_TD_
