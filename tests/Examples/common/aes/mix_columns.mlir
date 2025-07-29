// Mix columns step, computing a linear combination of the columns.

#map = affine_map<(d0) -> (d0)>

#mapA = affine_map<(d0, d1) -> (d0, d1)> // output of 2
#map1A = affine_map<(d0, d1) -> (d1)> // output dim 1
#map2A = affine_map<(d0, d1) -> (d0)> // output dim 1
module {
  func.func private @mul_gf256_2(%x: i8 {secret.secret}, %y: i8) -> (i8) {
      %z = arith.constant 0 : i8
      %c0 = arith.constant 0 : i8
      %c1 = arith.constant 1 : i8
      %2:2 = affine.for %4 = 0 to 2 iter_args(%5 = %x, %6 = %z) -> (i8, i8) {
        // z = if y & (1 << i): z^x else z
        %44 = arith.index_cast %4 : index to i8
        %ysh = arith.shrsi %y, %44 : i8
        %ysha = arith.andi %ysh, %c1 : i8
        %91 = arith.trunci %ysha : i8 to i1
        %10 = arith.xori %6, %5 : i8
        %argz = arith.select %91, %10, %6 : i8
        // check if MSB of x is set
        %14 = arith.cmpi sle, %5, %c0 : i8
        %11 = arith.constant 1 : i8
        %12 = arith.shli %5, %11 : i8
        %17 = arith.constant 27 : i8
        %122 = arith.xori %12, %17 : i8
        %argxx = arith.select %14, %122, %12 : i8
        affine.yield %argxx, %argz : i8, i8
      }
      func.return %2#1 : i8
  }

  func.func private @mix_single_column(%arg0: tensor<4xi8> {secret.secret}) -> tensor<4xi8> {
    // A 4x4 matrix with [2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]
    // use gf256 multiplication with the vector
    // use XOR to add
    %c0 = arith.constant 0 : i8
    %valA = arith.constant dense<[[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]]> : tensor<4x4xi8>
    %0 = tensor.splat %c0: tensor<4xi8>
    %1 = linalg.generic {indexing_maps = [#mapA, #map1A, #map2A], iterator_types = ["parallel", "reduction"]} ins(%valA, %arg0 : tensor<4x4xi8>, tensor<4xi8>) outs(%0 : tensor<4xi8>) {
    ^bb0(%in: i8, %in_0: i8, %out: i8):
      %1 = func.call @mul_gf256_2(%in_0, %in) : (i8, i8) -> i8
      %2 = arith.xori %out, %1 : i8
      linalg.yield %2 : i8
    } -> tensor<4xi8>
    return %1 : tensor<4xi8>
  }

  func.func @mix_columns(%arg0: tensor<16xi8> {secret.secret}) -> tensor<16xi8> {
    %out = tensor.empty() : tensor<16xi8>
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %mix = affine.for %i = 0 to 4 iter_args(%arg1 = %out) -> (tensor<16xi8>) {
      %index = arith.muli %i, %c4 : index
      %index1 = arith.addi %index, %c1 : index
      %index2 = arith.addi %index1, %c1 : index
      %index3 = arith.addi %index2, %c1 : index
      // extract_slice bufferizes to reinterpret_cast
      %extracted1 = tensor.extract %arg0[%index] : tensor<16xi8>
      %extracted2 = tensor.extract %arg0[%index1] : tensor<16xi8>
      %extracted3 = tensor.extract %arg0[%index2] : tensor<16xi8>
      %extracted4 = tensor.extract %arg0[%index3] : tensor<16xi8>
      %extracted = tensor.from_elements %extracted1, %extracted2, %extracted3, %extracted4 : tensor<4xi8>
      %mixed = func.call @mix_single_column(%extracted) : (tensor<4xi8>) -> tensor<4xi8>
      %emixed = tensor.extract %mixed[%c0] : tensor<4xi8>
      %emixed1 = tensor.extract %mixed[%c1] : tensor<4xi8>
      %emixed2 = tensor.extract %mixed[%c2] : tensor<4xi8>
      %emixed3 = tensor.extract %mixed[%c3] : tensor<4xi8>
      %inserted = tensor.insert %emixed into %arg1[%index] : tensor<16xi8>
      %inserted1 = tensor.insert %emixed1 into %inserted[%index1] : tensor<16xi8>
      %inserted2 = tensor.insert %emixed2 into %inserted1[%index2] : tensor<16xi8>
      %inserted3 = tensor.insert %emixed3 into %inserted2[%index3] : tensor<16xi8>
      affine.yield %inserted3 : tensor<16xi8>
    }
    return %mix : tensor<16xi8>
  }
}
