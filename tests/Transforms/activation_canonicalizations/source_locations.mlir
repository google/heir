// RUN: heir-opt %s --activation-canonicalizations --mlir-print-debuginfo | heir-opt --mlir-print-debuginfo | FileCheck %s
// CHECK-DAG: #[[PY:.*]] = loc("model.py":11:0)
// CHECK-DAG: #[[STACK:.*]] = loc(callsite({{.*}} at #[[PY]]))
// CHECK: arith.maximumf {{.*}} loc(#[[STACK]])
#loc1 = loc("torch/nn/modules/activation.py":143:0)
#loc2 = loc("model.py":11:0)
#map = affine_map<(d0, d1) -> (d0, d1)>
#loc3 = loc(callsite(#loc1 at #loc2))
module {
  func.func @main(%arg0: tensor<1x4xf32> loc(callsite(#loc1 at #loc2))) -> tensor<1x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32 loc(#loc)
    %0 = tensor.empty() : tensor<1x4xf32> loc(#loc3)
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x4xf32>) outs(%0 : tensor<1x4xf32>) {
    ^bb0(%in: f32 loc(callsite(#loc1 at #loc2)), %out: f32 loc(callsite(#loc1 at #loc2))):
      %2 = arith.cmpf ugt, %in, %cst : f32 loc(#loc3)
      %3 = arith.select %2, %in, %cst : f32 loc(#loc3)
      linalg.yield %3 : f32 loc(#loc3)
    } -> tensor<1x4xf32> loc(#loc3)
    return %1 : tensor<1x4xf32> loc(#loc3)
  } loc(#loc3)
} loc(#loc)
#loc = loc(unknown)
