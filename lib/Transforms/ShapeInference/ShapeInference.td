#ifndef LIB_TRANSFORMS_SHAPEINFERENCE_SHAPEINFERENCE_TD_
#define LIB_TRANSFORMS_SHAPEINFERENCE_SHAPEINFERENCE_TD_

include "mlir/Pass/PassBase.td"

def ShapeInference : Pass<"shape-inference"> {
  let summary = "Infer shapes for shaped types";

  let dependentDialects = [
  ];
}


#endif // LIB_TRANSFORMS_SHAPEINFERENCE_SHAPEINFERENCE_TD_
