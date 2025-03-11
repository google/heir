// RUN: heir-opt --verify-diagnostics --split-input-file %s


#row_major = #tensor_ext.layout<map = (d0, d1) -> (16*d0 + d1)>
#col_major = #tensor_ext.layout<map = (d0) -> (d0)>
func.func @test_convert_layout(%0: tensor<16x16xi32>) -> tensor<16x16xi32> {
  // expected-error@+1 {{requires tensor rank (after alignment) to match the layout map's dimension count but found rank 2 and layout #tensor_ext.layout<map = (d0) -> (d0)>}}
  %1 = tensor_ext.convert_layout %0 {from_layout = #row_major, to_layout = #col_major} : tensor<16x16xi32>
  return %1 : tensor<16x16xi32>
}

// -----

#row_major = #tensor_ext.layout<map = (d0, d1) -> (16*d0 + d1)>
#col_major = #tensor_ext.layout<map = (d0) -> (d0)>
func.func @test_convert_layout(%0: tensor<16x16xi32>) -> tensor<16x16xi32> {
  // expected-error@+1 {{op requires tensor rank (after alignment) to match the layout map's dimension count but found rank 2 and layout #tensor_ext.layout<map = (d0) -> (d0)>}}
  %1 = tensor_ext.convert_layout %0 {from_layout = #col_major, to_layout = #row_major} : tensor<16x16xi32>
  return %1 : tensor<16x16xi32>
}
