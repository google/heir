#ifndef LIB_TRANSFORMS_LAYOUTOPTIMIZATION_LAYOUTOPTIMIZATION_TD_
#define LIB_TRANSFORMS_LAYOUTOPTIMIZATION_LAYOUTOPTIMIZATION_TD_

include "mlir/Pass/PassBase.td"

def LayoutOptimization : Pass<"layout-optimization"> {
  let summary = "Optimize layout conversions in the IR";
  let description = [{
  This pass performance a greedy layout optimization pass similar to the
  automatic layout assignment from [A Tensor Compiler with Automatic Data
  Packing for Simple and Efficient Fully Homomorphic
  Encryption](https://dl.acm.org/doi/pdf/10.1145/3656382). The pass assumes that
  an initial layout assignment was provided on each operation through the
  `layout-propagation` pass.

  The pass iterates on each operation of the IR in reverse order, attempting to
  hoist a layout conversion of the operation's result before the operation. For
  each of the result's layout conversions, the pass will compute the net cost of
  hoisting the conversion through the operation by considering the following:
    1. The cost of performing the operation with new input layouts that result
       in the desired layout.
    2. The cost of the converting the layout of each input.
    3. The new cost of converting from the desired layout to each other layout
       conversions of the result.

  The layout conversion that results in the lowest net cost is chosen to be
  hoisted.

  Examples:

  The second layout conversion could be eliminated by performing the first
  addition operation under #map1.

  ```mlir
  !tensor = tensor<32xi16>
  !stensor = !secret.secret<!tensor>

  #map = affine_map<(d0) -> (d0 + 1 mod 32)>
  #map1 = affine_map<(d0) -> (d0)>
  module {
    func.func @push_conversion(%arg0: !stensor {tensor_ext.layout = #map}, %arg1: !stensor {tensor_ext.layout = #map1}, %arg2: !stensor {tensor_ext.layout = #map1}) -> (!stensor {tensor_ext.layout = #map}) {
      %0 = secret.generic(%arg0 : !stensor {tensor_ext.layout = #map}, %arg1 : !stensor {tensor_ext.layout = #map1}, %arg2 : !stensor {tensor_ext.layout = #map1}) {
      ^body(%input0: tensor<32xi16>, %input1: tensor<32xi16>, %input2: tensor<32xi16>):
        %1 = tensor_ext.convert_layout %input1 {from_layout = #map1, tensor_ext.layout = [#map], to_layout = #map} : tensor<32xi16>
        %2 = arith.addi %input0, %1 {tensor_ext.layout = #map} : tensor<32xi16>
        %3 = tensor_ext.convert_layout %2 {from_layout = #map, tensor_ext.layout = [#map1], to_layout = #map1} : tensor<32xi16>
        %4 = arith.addi %3, %input2 {tensor_ext.layout = #map1} : tensor<32xi16>
        secret.yield %4 : tensor<32xi16>
      } -> (!stensor {tensor_ext.layout = #map1})
      return %0 : !stensor
    }
  }
  ```

  This pass produces:

  ```mlir
  !tensor = tensor<32xi16>
  !stensor = !secret.secret<!tensor>

  #map = affine_map<(d0) -> (d0 + 1)>
  #map1 = affine_map<(d0) -> (d0)>
  module {
    func.func @push_conversion(%arg0: !stensor {tensor_ext.layout = #map1}, %arg1: !stensor {tensor_ext.layout = #map}, %arg2: !stensor {tensor_ext.layout = #map1}) -> (!stensor {tensor_ext.layout = #map}) {
      %0 = secret.generic(%arg0 : !stensor {tensor_ext.layout = #map}, %arg1 : !stensor {tensor_ext.layout = #map1}, %arg2 : !stensor {tensor_ext.layout = #map1}) {
      ^body(%input0: tensor<32xi16>, %input1: tensor<32xi16>, %input2: tensor<32xi16>):
        %1 = tensor_ext.convert_layout %input0 {from_layout = #map, tensor_ext.layout = #map1, to_layout = #map1} : tensor<32xi16>
        %2 = arith.addi %1, %input1 {tensor_ext.layout = #map1} : tensor<32xi16>
        %3 = arith.addi %2, %input2 {tensor_ext.layout = #map1} : tensor<32xi16>
        secret.yield %3 : tensor<32xi16>
      } -> (!stensor {tensor_ext.layout = #map1})
      return %0 : !stensor
    }
  }
  ```
  }];

  let dependentDialects = [
    "mlir::heir::tensor_ext::TensorExtDialect",
  ];
}

#endif  // LIB_TRANSFORMS_LAYOUTOPTIMIZATION_LAYOUTOPTIMIZATION_TD_
