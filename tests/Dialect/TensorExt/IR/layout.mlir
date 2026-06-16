// RUN: heir-opt --verify-diagnostics --split-input-file %s

// 1024 x 1024 matrix -> 1024 cts with 1024 slots each
// via halevi-shoup diagonal layout
#layout = #tensor_ext.layout<"{ [row, col] -> [ct, slot] : (slot mod 1024) - row = 0 and (ct + slot) mod 1024 - col = 0 and row >= 0 and col >= 0 and ct >= 0 and slot >= 0 }">
func.func private @test_fn(tensor<16xi32> {foo.bar = #layout})

// -----

#layout = #tensor_ext.layout<"{ [row, col] -> [ct, slot] : (slot - row) mod 16 = 0 and (ct + slot - col) mod 16 = 0 and row >= 0 and col >= 0 and ct >= 0 and slot >= 0 and 31 >= slot and 15 >= ct and 15 >= row and 15 >= col }">

func.func @test_empty_array(%arg0: tensor<16x16xi32>) -> tensor<16x16xi32> {
  // expected-error @+1 {{'tensor_ext.assign_layout' op layout array cannot be empty}}
  %0 = tensor_ext.assign_layout %arg0 {layout = []} : tensor<16x16xi32>
  return %0 : tensor<16x16xi32>
}

// -----

#layout = #tensor_ext.layout<"{ [row, col] -> [ct, slot] : (slot - row) mod 16 = 0 and (ct + slot - col) mod 16 = 0 and row >= 0 and col >= 0 and ct >= 0 and slot >= 0 and 31 >= slot and 15 >= ct and 15 >= row and 15 >= col }">

func.func @test_first_not_layout_attr(%arg0: tensor<16x16xi32>) -> tensor<16x16xi32> {
  // expected-error @+1 {{'tensor_ext.assign_layout' op attribute 'layout' failed to satisfy constraint}}
  %0 = tensor_ext.assign_layout %arg0 {layout = ["bad_attr"]} : tensor<16x16xi32>
  return %0 : tensor<16x16xi32>
}

// -----

#layout = #tensor_ext.layout<"{ [row, col] -> [ct, slot] : (slot - row) mod 16 = 0 and (ct + slot - col) mod 16 = 0 and row >= 0 and col >= 0 and ct >= 0 and slot >= 0 and 31 >= slot and 15 >= ct and 15 >= row and 15 >= col }">

func.func @test_non_first_not_layout_attr(%arg0: tensor<16x16xi32>) -> tensor<16x16xi32> {
  // expected-error @+1 {{'tensor_ext.assign_layout' op attribute 'layout' failed to satisfy constraint}}
  %0 = tensor_ext.assign_layout %arg0 {layout = [#layout, "bad_attr"]} : tensor<16x16xi32>
  return %0 : tensor<16x16xi32>
}

// -----

#layout1 = #tensor_ext.layout<"{ [row, col] -> [o0, o1, o2] : o0 = row and o1 = col and o2 = col and 0 <= row <= 15 and 0 <= col <= 15 }">
#layout2 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = i0 and slot = i1 and 0 <= i0 <= 15 and 0 <= i1 <= 15 and 0 <= slot <= 31 }">

func.func @test_dim_mismatch(%arg0: tensor<16x16xi32>) -> tensor<16x16xi32> {
  // expected-error @+1 {{'tensor_ext.assign_layout' op layout 1 domain size (2) must match layout 0 range size (3)}}
  %0 = tensor_ext.assign_layout %arg0 {layout = [#layout1, #layout2]} : tensor<16x16xi32>
  return %0 : tensor<16x16xi32>
}

// -----

#layout = #tensor_ext.layout<"{ [row, col] -> [ct, slot] : (slot - row) mod 16 = 0 and (ct + slot - col) mod 16 = 0 and row >= 0 and col >= 0 and ct >= 0 and slot >= 0 and 31 >= slot and 15 >= ct and 15 >= row and 15 >= col }">

func.func @test_rank_mismatch(%arg0: tensor<16xi32>) -> tensor<16xi32> {
  // expected-error @+1 {{'tensor_ext.assign_layout' op requires tensor rank to match the layout's domain size, but found rank 1 and domain size 2}}
  %0 = tensor_ext.assign_layout %arg0 {layout = [#layout]} : tensor<16xi32>
  return %0 : tensor<16xi32>
}
