#ifndef LIB_TRANSFORMS_TENSORTOSCALARS_TENSORTOSCALARS_TD_
#define LIB_TRANSFORMS_TENSORTOSCALARS_TENSORTOSCALARS_TD_

include "mlir/Pass/PassBase.td"


def TensorToScalars : Pass<"convert-tensor-to-scalars"> {

  let summary = "Effectively 'unrolls' tensors of static shape to scalars.";

  let description = [{
    This pass will convert a static-shaped tensor type to a TypeRange
    containing product(dim) copies of the element type of the tensor.
    This pass currently includes two patterns:

    1. It converts tensor.from_elements operations to
    the corresponding scalar inputs.
    2. It converts tensor.insert operations by updating the
    ValueRange corresponding to the converted input and
    updating it with the scalar to be inserted.

    It also applies folders greedily to simplify, e.g., extract(from_elements).

    Note: The pass is designed to be run on an IR, where the only operations
    with tensor typed operands are tensor "management" operations such as insert/extract,
    with all other operations (e.g., arith operations) already taking (extracted) scalar inputs.
    For example, an IR where elementwise operations have been converted to scalar operations via
    `--convert-elementwise-to-affine`.

    The pass might insert new tensor.from_elements operations or manually create the scalar ValueRange
    via inserting tensor.extract operations if any operations remain that operate on tensors.
    The pass currently applies irrespective of tensor size, i.e., might be very slow for large tensors.

    TODO (#1023): Extend this pass to support more tensor operations, e.g., `tensor.slice`
  }];

  let dependentDialects = [
    "tensor::TensorDialect"
  ];

  let options = [
    Option<"maxSize", "max-size", "int", /*default=*/"8",
           "Limits `unrolling` to tensors with at most max-size elements">,
  ];
}

#endif  // LIB_TRANSFORMS_TENSORTOSCALARS_TENSORTOSCALARS_TD_
