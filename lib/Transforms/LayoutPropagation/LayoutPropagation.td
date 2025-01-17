#ifndef LIB_TRANSFORMS_LAYOUTPROPAGATION_LAYOUTPROPAGATION_TD_
#define LIB_TRANSFORMS_LAYOUTPROPAGATION_LAYOUTPROPAGATION_TD_

include "mlir/Pass/PassBase.td"

def LayoutPropagation : Pass<"layout-propagation"> {
  let summary = "Propagate ciphertext layouts through the IR";
  let description = [{
  This pass performs a forward propagation of layout (packing) information
  through the input IR, starting from the assumption that each function
  argument tensor has a row-major layout.

  The chosen layouts (`affine_map`s) are annotated on ops throughout the IR. In particular,

  - Ops with a nested region and block arguments use a dictionary attribute to
    mark the layout of each block argument. `func.func` in particular uses the
    `tensor_ext.layout` dialect attribute, while others use an affine map
    attribute.
  - Other ops annotate their results with layouts as an ArrayAttr of affine
    maps. The order of the affine maps corresponds to the order of results.

  When two incompatible layouts are encountered during the propagation, a
  `tensor_ext.convert_layout` op is inserted. For an example of two
  incompatible layouts, consider the `tensor_ext.sum` operation. Summing along
  each of the two axes of a row-major-packed `tensor<32x32xi16>` results in two
  `tensor<32xi16>`, but with incompatible layouts: the first has a compact
  layout residing in the first 32-entries of a ciphertext, while the second is
  a strided layout with a stride of 32.

  The converted op is arbitrarily chosen to have the layout of the first input,
  and later passes are responsible for optimizing the choice of which operand
  is converted and where the conversion operations are placed. This separation
  of duties allows this pass to be reused as a pure dataflow analysis, in which
  case it annotates an un-annotated IR with layout attributes.

  Examples:

  Two incompatible summations require a layout conversion

  ```mlir
  func.func @incompatible_summed_dims(
        %arg0: !secret.secret<tensor<32x32xi16>>,
        %arg1: !secret.secret<tensor<32x32xi16>>
  ) -> !secret.secret<tensor<32xi16>> {
    %0 = secret.generic ins(
      %arg0, %arg1 : !secret.secret<tensor<32x32xi16>>, !secret.secret<tensor<32x32xi16>>) {
    ^bb0(%input0: tensor<32x32xi16>, %input1: tensor<32x32xi16>):
      %1 = tensor_ext.sum %input0, 0 : tensor<32x32xi16> -> tensor<32xi16>
      %2 = tensor_ext.sum %input1, 1 : tensor<32x32xi16> -> tensor<32xi16>
      %3 = arith.addi %1, %2 : tensor<32xi16>
      secret.yield %3 : tensor<32xi16>
    } -> !secret.secret<tensor<32xi16>>
    return %0 : !secret.secret<tensor<32xi16>>
  }
  ```

  This pass produces:

  ```mlir
  #map = affine_map<(d0, d1) -> (d0 * 32 + d1)>
  #map1 = affine_map<(d0) -> (d0)>
  #map2 = affine_map<(d0) -> (d0 * 32)>

  func.func @incompatible_summed_dims(
    %arg0: !secret.secret<tensor<32x32xi16>>
      {tensor_ext.layout = #tensor_ext.layout<layout = (d0, d1) -> (d0 * 32 + d1)>},
    %arg1: !secret.secret<tensor<32x32xi16>>
      {tensor_ext.layout = #tensor_ext.layout<layout = (d0, d1) -> (d0 * 32 + d1)>}
  ) -> !secret.secret<tensor<32xi16>> {
    %0 = secret.generic ins(
        %arg0, %arg1 : !secret.secret<tensor<32x32xi16>>, !secret.secret<tensor<32x32xi16>>)
        attrs = {
          arg0 = {layout = #map},
          arg1 = {layout = #map},
          layout = [#map1]
        } {
    ^bb0(%input0: tensor<32x32xi16>, %input1: tensor<32x32xi16>):
      %1 = tensor_ext.sum %input0, 0 {layout = [#map1]} : tensor<32x32xi16> -> tensor<32xi16>
      %2 = tensor_ext.sum %input1, 1 {layout = [#map2]} : tensor<32x32xi16> -> tensor<32xi16>
      %3 = tensor_ext.convert_layout %2 {
          from_layout = #map2,
          to_layout = #map1,
          layout = [#map1]} : tensor<32xi16>
      %4 = arith.addi %1, %3 {layout = [#map1]} : tensor<32xi16>
      secret.yield %4 : tensor<32xi16>
    } -> !secret.secret<tensor<32xi16>>
    return %0 : !secret.secret<tensor<32xi16>>
  }
  ```
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::heir::secret::SecretDialect",
    "mlir::heir::tensor_ext::TensorExtDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_TRANSFORMS_LAYOUTPROPAGATION_LAYOUTPROPAGATION_TD_
