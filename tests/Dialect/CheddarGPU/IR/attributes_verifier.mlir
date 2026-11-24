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

// -----
// ParamsAttr primes must be positive

module attributes {
  // expected-error @below {{'ter' prime 0 is outside the range (0, 2^31)}}
  cheddar_gpu.params = #cheddar_gpu.params<word = 32, log_n = 16, ter = [0], main = [2000, 3000], aux = [4000], level_config = [[[2 : i32, 1 : i32]]], alpha = 1, dnum = 0>
} {}

// -----

module attributes {
  // expected-error @below {{'aux' prime -5 is outside the range (0, 2^63)}}
  cheddar_gpu.params = #cheddar_gpu.params<word = 64, log_n = 16, ter = [1000], main = [2000, 3000], aux = [-5], level_config = [[[2 : i32, 1 : i32]]], alpha = 1, dnum = 0>
} {}

// -----
// ParamsAttr primes must leave the top bit of the word unused

module attributes {
  // expected-error @below {{'main' prime 2147483648 is outside the range (0, 2^31)}}
  cheddar_gpu.params = #cheddar_gpu.params<word = 32, log_n = 16, ter = [1000], main = [2000, 2147483648], aux = [4000], level_config = [[[2 : i32, 1 : i32]]], alpha = 1, dnum = 0>
} {}

// -----
// ParamsAttr primes must fit in the word

module attributes {
  // expected-error @below {{'aux' prime 4294967311 is outside the range (0, 2^31)}}
  cheddar_gpu.params = #cheddar_gpu.params<word = 32, log_n = 16, ter = [1000], main = [2000, 3000], aux = [4294967311], level_config = [[[2 : i32, 1 : i32]]], alpha = 1, dnum = 0>
} {}

// -----
// ParamsAttr accepts primes up to 2^(word-1) - 1

module attributes {
  cheddar_gpu.params = #cheddar_gpu.params<word = 32, log_n = 16, ter = [1000], main = [2000, 2147483647], aux = [4000], level_config = [[[2 : i32, 1 : i32]]], alpha = 1, dnum = 0>
} {}

// -----

module attributes {
  cheddar_gpu.params = #cheddar_gpu.params<word = 64, log_n = 16, ter = [1000], main = [2147483648, 9223372036854775807], aux = [4000], level_config = [[[2 : i32, 1 : i32]]], alpha = 1, dnum = 0>
} {}
