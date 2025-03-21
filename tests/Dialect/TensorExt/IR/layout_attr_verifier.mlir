// RUN: heir-opt --verify-diagnostics --split-input-file %s

#align = #tensor_ext.alignment<in = [16], out = [32], padding = [16], paddingValue = 0:i32>
// expected-error@below {{The affine map's input size must match the number of dimensions of alignment.out}}
#layout = #tensor_ext.layout<alignment = #align, map = (d0, d1) -> (d0 + d1)>
func.func private @test_fn(tensor<16xi32> {foo.bar = #layout})

// -----

#align = #tensor_ext.alignment<in = [16], out = [32, 32], insertedDims = [0], padding = [0, 16], paddingValue = 0:i32>
// expected-error@below {{The affine map's input size must match the number of dimensions of alignment.out}}
#layout = #tensor_ext.layout<alignment = #align, map = (d0) -> (d0)>
func.func private @test_fn(tensor<16xi32> {foo.bar = #layout})
