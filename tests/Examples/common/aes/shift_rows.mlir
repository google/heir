// Shift rows function, shifting the 4 rows by 0, 1, 2, and 3 steps to the left.

#map = affine_map<(d0) -> (d0)>
module {
  func.func @shift_rows(%arg0: tensor<16xi8> {secret.secret}) -> tensor<16xi8> {
    %0 = tensor.empty() : tensor<16xi8>
    %indices = arith.constant dense<[0, 1, 2, 3, 5, 6, 7, 4, 10, 11, 8, 9, 15, 12, 13, 14]> : tensor<16xindex>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%indices : tensor<16xindex>) outs(%0 : tensor<16xi8>) {
    ^bb0(%in: index, %out: i8):
      %extracted = tensor.extract %arg0[%in] : tensor<16xi8>
      linalg.yield %extracted : i8
    } -> tensor<16xi8>
    return %1 : tensor<16xi8>
  }
}
