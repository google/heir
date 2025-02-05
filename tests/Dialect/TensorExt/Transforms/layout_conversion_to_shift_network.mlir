// RUN: heir-opt --layout-conversion-to-shift-network %s | FileCheck %s

#map1 = affine_map<(d0) -> (d0 + 4 mod 64)>
#map2 = affine_map<(d0) -> (3 * d0 mod 64)>
func.func @test_convert_layout(%0: tensor<64xi32>) -> tensor<64xi32> {
  %1 = tensor_ext.convert_layout %0 {from_layout = #map1, to_layout = #map2} : tensor<64xi32>
  return %1 : tensor<64xi32>
}
