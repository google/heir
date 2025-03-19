// RUN: heir-opt --verify-diagnostics --split-input-file %s

// expected-error@below {{in and out may not be empty arrays}}
#align = #tensor_ext.alignment<in = [], out = [32]>
func.func private @test_fn(tensor<16xi32> {foo.bar = #align})

// -----

// expected-error@below {{in and out may not be empty arrays}}
#align = #tensor_ext.alignment<in = [16], out = []>
func.func private @test_fn(tensor<16xi32> {foo.bar = #align})

// -----

// expected-error@below {{paddingValue must be set if padding is set}}
#align = #tensor_ext.alignment<in = [16], out = [32], padding = [16]>
func.func private @test_fn(tensor<16xi32> {foo.bar = #align})

// -----

// expected-error@below {{in.size() + insertedDims.size() must equal out.size()}}
#align = #tensor_ext.alignment<in = [16], out = [32], insertedDims = [1], padding = [16], paddingValue = 0:i32>
func.func private @test_fn(tensor<16xi32> {foo.bar = #align})

// -----

// expected-error@below {{padding.size() must equal out.size()}}
#align = #tensor_ext.alignment<in = [16], out = [32], insertedDims = [], padding = [1, 2], paddingValue = 0:i32>
func.func private @test_fn(tensor<16xi32> {foo.bar = #align})

// -----

// expected-error@below {{insertedDims must all be unique}}
#align = #tensor_ext.alignment<in = [16], out = [16, 1, 1, 1, 1, 1], insertedDims = [1, 2, 3, 3, 4]>
func.func private @test_fn(tensor<16xi32> {foo.bar = #align})

// -----

// expected-error@below {{After inserting dims and padding, each axis must have size dividing or divisible by the corresponding output axis size}}
#align = #tensor_ext.alignment<in = [16], out = [32], padding = [3], paddingValue = 0:i32>
func.func private @test_fn(tensor<16xi32> {foo.bar = #align})
