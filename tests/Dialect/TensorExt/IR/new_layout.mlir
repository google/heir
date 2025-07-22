// RUN: heir-opt --verify-diagnostics --split-input-file %s

// 1024 x 1024 matrix -> 1024 cts with 1024 slots each
// via halevi-shoup diagonal layout
#layout = #tensor_ext.new_layout<
  domainSize=2,
  relation="(row, col, ct, slot) : ((slot mod 1024) - row == 0, (ct + slot) mod 1024 - col == 0, row >= 0, col >= 0, ct >= 0, slot >= 0)"
>
func.func private @test_fn(tensor<16xi32> {foo.bar = #layout})

// -----
