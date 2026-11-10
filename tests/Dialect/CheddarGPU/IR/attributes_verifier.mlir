// RUN: heir-opt --verify-diagnostics --split-input-file %s

// ParamsAttr word must be 32 or 64

module attributes {
  // expected-error @below {{word must be 32 or 64, but got 16}}
  cheddar_gpu.params = #cheddar_gpu.params<word = 16, log_n = 16, ter = [1000], main = [2000, 3000], aux = [4000], level_config = [[[2 : i32, 1 : i32]]], alpha = 1, dnum = 0>
} {}

// -----
// ParamsAttr log_n must be positive

module attributes {
  // expected-error @below {{log_n must be positive, but got 0}}
  cheddar_gpu.params = #cheddar_gpu.params<word = 32, log_n = 0, ter = [1000], main = [2000, 3000], aux = [4000], level_config = [[[2 : i32, 1 : i32]]], alpha = 1, dnum = 0>
} {}

// -----
// ParamsAttr alpha must be positive

module attributes {
  // expected-error @below {{alpha must be positive, but got 0}}
  cheddar_gpu.params = #cheddar_gpu.params<word = 32, log_n = 16, ter = [1000], main = [2000, 3000], aux = [4000], level_config = [[[2 : i32, 1 : i32]]], alpha = 0, dnum = 0>
} {}
