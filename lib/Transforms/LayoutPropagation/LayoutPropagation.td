#ifndef LIB_TRANSFORMS_LAYOUTPROPAGATION_LAYOUTPROPAGATION_TD_
#define LIB_TRANSFORMS_LAYOUTPROPAGATION_LAYOUTPROPAGATION_TD_

include "mlir/Pass/PassBase.td"

def LayoutPropagation : Pass<"layout-propagation"> {
  let summary = "Propagate ciphertext layouts through the IR";
  let description = [{
  This pass performs a forward propagation of layout (packing) information
  through the input IR, starting from the assumption that each secret tensor
  argument to a function has a row-major layout.

  The chosen layouts (Integer Relations) are annotated on ops throughout the IR.
  In particular,

  - Ops with a nested region and block arguments use a dictionary attribute to
    mark the layout of each block argument. `func.func` in particular uses the
    `tensor_ext.layout` dialect attribute, while others use an affine map
    attribute.
  - Other ops annotate their results with layouts as an ArrayAttr of layouts.
    The order of the affine maps corresponds to the order of results.

  When a plaintext SSA value is encountered as an input to a secret operation,
  a `tensor_ext.assign_layout` op is inserted that assigns it a default layout.
  This semantically corresponds to a plaintext packing operation. This is
  performed as late as possible before the SSA value is used, to avoid
  unnecessary layout conversions of plaintexts. This implies that not all SSA
  values in the IR are annotated with layouts, only those that have secret
  results or secret operands.

  When two incompatible layouts are encountered as operands to the same op,
  `tensor_ext.convert_layout` ops are inserted. For example, consider the
  `linalg.reduce` operation for a summation. Summing along each of the two axes
  of a row-major-packed `tensor<32x32xi16>` results in two `tensor<32xi16>`,
  but with incompatible layouts: the first has a compact layout residing in the
  first 32-entries of a ciphertext, while the second is a strided layout with a
  stride of 32.

  The converted op is arbitrarily chosen to have the layout of the first input,
  and later passes are responsible for optimizing the choice of which operand
  is converted and where the conversion operations are placed. This separation
  of duties allows this pass to be reused as a pure dataflow analysis, in which
  case it annotates an un-annotated IR with layout attributes.

  Secret structured ops without a dedicated kernel are first generalized to
  `linalg.generic`, whose cleartext inputs get no layout. For a
  `linalg.generic`, a partial write (e.g. a piece of a concatenation) keeps the
  layout of its init; otherwise each result element is kept in the slots of
  the secret input element read in the same iteration (for reductions, the
  first of them), falling back to the init's or the default layout. An
  explicit `tensor_ext.layout` attribute on the op overrides this.

  (* example filepath=tests/Transforms/layout_propagation/doctest.mlir *)
  }];
  let options = [
    Option<
      "minSlotCount",
      "min-slot-count",
      "int",
      /*default=*/"1024",
      "Minimum number of slots to use for packing messages in ciphertexts."
    >
  ];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::heir::secret::SecretDialect",
    "mlir::heir::tensor_ext::TensorExtDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_TRANSFORMS_LAYOUTPROPAGATION_LAYOUTPROPAGATION_TD_
