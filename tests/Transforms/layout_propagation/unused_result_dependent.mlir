// RUN: heir-opt --layout-propagation=min-slot-count=32 --verify-diagnostics %s

// Only the first result is used, but it reads the second accumulator, which
// reads the third, so none of the results can be dropped and the generic
// kernel rejects the op.

func.func @chained_accumulators(%arg0: !secret.secret<tensor<2x16xf32>>) -> !secret.secret<tensor<2xf32>> {
  %cst = arith.constant dense<0.0> : tensor<2xf32>
  %r = secret.generic(%arg0: !secret.secret<tensor<2x16xf32>>) {
  ^body(%x: tensor<2x16xf32>):
    // expected-error@+1 {{the generic FHE kernel requires a single statically shaped result}}
    %4:3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%x : tensor<2x16xf32>) outs(%cst, %cst, %cst : tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) {
    ^bb0(%in: f32, %a: f32, %b: f32, %c: f32):
      %5 = arith.addf %a, %b : f32
      %6 = arith.addf %b, %c : f32
      %7 = arith.addf %c, %in : f32
      linalg.yield %5, %6, %7 : f32, f32, f32
    } -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)
    secret.yield %4#0 : tensor<2xf32>
  } -> !secret.secret<tensor<2xf32>>
  return %r : !secret.secret<tensor<2xf32>>
}
