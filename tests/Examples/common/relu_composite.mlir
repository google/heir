#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  // A ReLU in the shape the torch importer produces: a linalg.generic carrying
  // the calibrated domain, with cmpf+select in its body. This exercises the
  // whole composite-sign path end to end:
  //   - activation-canonicalizations rewrites select -> arith.maximumf and
  //     forwards domain_lower/domain_upper onto it,
  //   - polynomial-approximation (use-composite-relu) turns that into
  //     x * step(x/B) with three chained Chebyshev evals, B = max|domain|.
  func.func @relu_composite(%arg0: tensor<1x16xf32> {secret.secret}) -> tensor<1x16xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x16xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"], domain_lower = -2.0733445882797241 : f64, domain_upper = 2.0503503084182739 : f64} ins(%arg0 : tensor<1x16xf32>) outs(%0 : tensor<1x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.cmpf ugt, %in, %cst : f32
      %3 = arith.select %2, %in, %cst : f32
      linalg.yield %3 : f32
    } -> tensor<1x16xf32>
    return %1 : tensor<1x16xf32>
  }
}
