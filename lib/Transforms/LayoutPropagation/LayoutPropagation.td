#ifndef LIB_TRANSFORMS_LAYOUTPROPAGATION_LAYOUTPROPAGATION_TD_
#define LIB_TRANSFORMS_LAYOUTPROPAGATION_LAYOUTPROPAGATION_TD_

include "mlir/Pass/PassBase.td"

def LayoutPropagation : Pass<"layout-propagation"> {
  let summary = "Propagate ciphertext layouts through the IR";
  let description = [{
  This pass performs a forward propagation of layout (packing) information
  through the input IR, starting from the assumption that each secret tensor
  argument to a function has a row-major layout.

  The chosen layouts (`affine_map`s) are annotated on ops throughout the IR.
  In particular,

  - Ops with a nested region and block arguments use a dictionary attribute to
    mark the layout of each block argument. `func.func` in particular uses the
    `tensor_ext.layout` dialect attribute, while others use an affine map
    attribute.
  - Other ops annotate their results with layouts as an ArrayAttr of affine
    maps. The order of the affine maps corresponds to the order of results.

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

  Examples:

  Two incompatible summations require a layout conversion

  ```mlir
  !tensor = tensor<32x32xi16>
  !tensor2 = tensor<32xi16>
  !stensor = !secret.secret<!tensor>
  !stensor2 = !secret.secret<!tensor2>

  func.func @insert_conversion(%arg0: !stensor, %arg1: !stensor) -> !stensor2 {
    %out_1 = arith.constant dense<0> : !tensor2
    %out_2 = arith.constant dense<0> : !tensor2

    %0 = secret.generic ins(%arg0, %arg1: !stensor, !stensor) {
    ^body(%pt_arg0: !tensor, %pt_arg1: !tensor):
      %1 = linalg.reduce { arith.addi } ins(%pt_arg0:!tensor) outs(%out_1:!tensor2) dimensions = [0]
      %2 = linalg.reduce { arith.addi } ins(%pt_arg1:!tensor) outs(%out_2:!tensor2) dimensions = [1]
      %3 = arith.addi %1, %2 : !tensor2
      secret.yield %3 : !tensor2
    } -> !stensor2
    return %0 : !stensor2
  }
  ```

  This pass produces:

  ```mlir
  #map = affine_map<(d0, d1) -> (d0 * 32 + d1)>
  #map1 = affine_map<(d0) -> (d0)>
  #map2 = affine_map<(d0) -> (d0 * 32)>
  module {
    func.func @insert_conversion(
          %arg0: !secret.secret<tensor<32x32xi16>> {
              tensor_ext.layout = #tensor_ext.layout<layout = (d0, d1) -> (d0 * 32 + d1)>},
          %arg1: !secret.secret<tensor<32x32xi16>> {
              tensor_ext.layout = #tensor_ext.layout<layout = (d0, d1) -> (d0 * 32 + d1)>}
        ) -> (!secret.secret<tensor<32xi16>> {tensor_ext.layout = #tensor_ext.layout<layout = (d0) -> (d0)>}) {
      %cst = arith.constant dense<0> : tensor<32xi16>
      %cst_0 = arith.constant dense<0> : tensor<32xi16>
      %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<tensor<32x32xi16>>, !secret.secret<tensor<32x32xi16>>)
                          attrs = {arg0 = {tensor_ext.layout = #map}, arg1 = {tensor_ext.layout = #map}, layout = [#map1]} {
      ^body(%input0: tensor<32x32xi16>, %input1: tensor<32x32xi16>):
        %1 = tensor_ext.assign_layout %cst {tensor_ext.layout = #map1} : tensor<32xi16>
        %reduced = linalg.reduce { arith.addi {overflowFlags = #arith.overflow<none>} }
                    ins(%input0 : tensor<32x32xi16>)
                    outs(%1 : tensor<32xi16>)
                    dimensions = [0]  {tensor_ext.layout = [#map1]}

        %2 = tensor_ext.assign_layout %cst_0 {tensor_ext.layout = #map1} : tensor<32xi16>
        %3 = tensor_ext.convert_layout %2 {from_layout = #map1, layout = [#map2], to_layout = #map2} : tensor<32xi16>
        %reduced_1 = linalg.reduce { arith.addi {overflowFlags = #arith.overflow<none>} }
                    ins(%input1 : tensor<32x32xi16>)
                    outs(%3 : tensor<32xi16>)
                    dimensions = [1]  {tensor_ext.layout = [#map2]}

        %4 = tensor_ext.convert_layout %reduced_1 {from_layout = #map2, layout = [#map1], to_layout = #map1} : tensor<32xi16>
        %5 = arith.addi %reduced, %4 {tensor_ext.layout = [#map1]} : tensor<32xi16>
        secret.yield %5 : tensor<32xi16>
      } -> !secret.secret<tensor<32xi16>>
      return %0 : !secret.secret<tensor<32xi16>>
    }
  }
  ```
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

#endif  // LIB_TRANSFORMS_LAYOUTPROPAGATION_LAYOUTPROPAGATION_TD_
