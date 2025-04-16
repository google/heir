// Inverse mix columns step, computing a linear combination of the columns.

#map = affine_map<(d0) -> (d0)>

#mapA = affine_map<(d0, d1) -> (d0, d1)> // output of 2
#map1A = affine_map<(d0, d1) -> (d1)> // output dim 1
#map2A = affine_map<(d0, d1) -> (d0)> // output dim 1
module {
  func.func private @mul_gf256_4(%x: i8 {secret.secret}, %y: i8) -> (i8) {
      %z = arith.constant 0 : i8
      %2:2 = affine.for %4 = 0 to 4 iter_args(%5 = %x, %6 = %z) -> (i8, i8) {
        %7 = arith.constant 1 : i8
        %44 = arith.index_cast %4 : index to i8
        %8 = arith.shli %7, %44 : i8
        %9 = arith.andi %y, %8 : i8
        %91 = arith.trunci %9 : i8 to i1
        %10 = arith.xori %6, %5 : i8
        %argz = arith.select %91, %10, %6 : i8
        %11 = arith.constant 1 : i8
        %12 = arith.shli %argz, %11 : i8
        %13 = arith.constant 0x80 : i8
        %14 = arith.cmpi sge, %12, %13 : i8
        %argxx = scf.if %14 -> (i8) {
          %15 = arith.constant 255 : i8
          %16 = arith.andi %12, %15 : i8
          %17 = arith.constant 27 : i8
          %122 = arith.xori %16, %17 : i8
          scf.yield %122 : i8
        } else {
          scf.yield %12 : i8
        }
        affine.yield %argxx, %argz : i8, i8
      }
      func.return %2#1 : i8
  }

  func.func private @mix_single_column(%arg0: tensor<4xi8> {secret.secret}) -> tensor<4xi8> {
    // A 4x4 matrix with [2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]
    // use gf256 multiplication with the vector
    // use XOR to add
    %valA = arith.constant dense<[[0x0e, 0x0b, 0x0d, 0x09], [0x09, 0x0e, 0x0b, 0x0d], [0x0d, 0x09, 0x0e, 0x0b], [0x0b, 0x0d, 0x09, 0x0e]]> : tensor<4x4xi8>
    %0 = tensor.empty() : tensor<4xi8>
    %1 = linalg.generic {indexing_maps = [#mapA, #map1A, #map2A], iterator_types = ["parallel", "reduction"]} ins(%valA, %arg0 : tensor<4x4xi8>, tensor<4xi8>) outs(%0 : tensor<4xi8>) {
    ^bb0(%in: i8, %in_0: i8, %out: i8):
      %1 = func.call @mul_gf256_4(%in_0, %in) : (i8, i8) -> i8
      %2 = arith.xori %out, %1 : i8
      linalg.yield %2 : i8
    } -> tensor<4xi8>
    return %1 : tensor<4xi8>
  }

  func.func @mix_columns(%arg0: tensor<16xi8> {secret.secret}) -> tensor<16xi8> {
    %out = tensor.empty() : tensor<16xi8>
    %c4 = arith.constant 4 : index
    %mix = affine.for %i = 0 to 4 iter_args(%arg1 = %out) -> (tensor<16xi8>) {
      %index = arith.muli %i, %c4 : index
      %extracted = tensor.extract_slice %arg0[%index][4][1] : tensor<16xi8> to tensor<4xi8>
      %mixed = func.call @mix_single_column(%extracted) : (tensor<4xi8>) -> tensor<4xi8>
      %inserted = tensor.insert_slice %mixed into %arg1[%index][4][1] : tensor<4xi8> into tensor<16xi8>
      affine.yield %inserted : tensor<16xi8>
    }
    return %mix : tensor<16xi8>
  }
}
