#ifndef LIB_TRANSFORMS_SHAPEINFERENCE_SHAPEINFERENCE_TD_
#define LIB_TRANSFORMS_SHAPEINFERENCE_SHAPEINFERENCE_TD_

include "mlir/Pass/PassBase.td"

def ShapeInference : Pass<"shape-inference"> {
  let summary = "Infer shapes for shaped types";

  let description = [{
    This pass infers the shapes of shaped types in a function,
    starting from function arguments annotated with a {shape.shape} attribute.
    Shape inference is only supported for operations that implement InferTypeOpInterface.

    This is primarily intended to be used in conjunction with the Python frontend,
    which infers the rank, but not the length of each dimension, for tensor types.
  }];

  let dependentDialects = [
  ];
}


#endif // LIB_TRANSFORMS_SHAPEINFERENCE_SHAPEINFERENCE_TD_
