#ifndef LIB_TRANSFORMS_LAYOUTPROPAGATION_NEWLAYOUTPROPAGATION_TD_
#define LIB_TRANSFORMS_LAYOUTPROPAGATION_NEWLAYOUTPROPAGATION_TD_

include "mlir/Pass/PassBase.td"

def NewLayoutPropagation : Pass<"new-layout-propagation"> {
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

  TODO(#2047): Add doctest examples with the new layout
  }];
  let options = [
    Option<
      "ciphertextSize",
      "ciphertext-size",
      "int",
      /*default=*/"1024",
      "Power of two length of the ciphertexts the data is packed in."
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

#endif  // LIB_TRANSFORMS_LAYOUTPROPAGATION_NEWLAYOUTPROPAGATION_TD_
