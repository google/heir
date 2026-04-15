// RUN: heir-opt --verify-diagnostics --split-input-file %s

// expected-error @below {{`size` must be > 0, got 0}}
func.func private @bad_size(tensor<16xi32> {foo = #rotom.dim<dim = 0, size = 0, stride = 1>})

// -----

// expected-error @below {{`stride` must be > 0, got 0}}
func.func private @bad_stride(tensor<16xi32> {foo = #rotom.dim<dim = 0, size = 8, stride = 0>})

// -----

// expected-error @below {{`dim` must be >= -2, got -3}}
func.func private @bad_dim(tensor<16xi32> {foo = #rotom.dim<dim = -3, size = 8, stride = 1>})
