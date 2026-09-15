// RUN: heir-opt --verify-diagnostics --split-input-file %s

// expected-error @below {{`size` must be > 0, got 0}}
func.func private @bad_size(tensor<16xi32> {foo = #rotom.dim<[0:0:1]>})

// -----

// expected-error @below {{`stride` must be > 0, got 0}}
func.func private @bad_stride(tensor<16xi32> {foo = #rotom.dim<[0:8:0]>})

// -----

// expected-error @below {{`dim` must be >= -3, got -4}}
func.func private @bad_dim(tensor<16xi32> {foo = #rotom.dim<[-4:8:1]>})
