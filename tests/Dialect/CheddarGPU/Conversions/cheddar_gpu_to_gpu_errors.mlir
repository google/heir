// RUN: heir-opt %s --cheddar-gpu-to-gpu --verify-diagnostics --split-input-file

#eval = #cheddar_gpu.form<eval>

module {
  func.func @test_missing_params(%out: memref<4x65536xi32>, %lhs: memref<4x65536xi32>, %rhs: memref<4x65536xi32>) {
    // expected-error @below {{missing cheddar_gpu.params on module}}
    // expected-error @below {{failed to legalize operation 'cheddar_gpu.add'}}
    cheddar_gpu.add %out, %lhs, %rhs <{
      lhs_form = #eval,
      rhs_form = #eval,
      out_form = #eval
    }> : memref<4x65536xi32>
    return
  }
}

// -----

#eval = #cheddar_gpu.form<eval>
#params = #cheddar_gpu.params<word = 32, log_n = 6, ter = [17], main = [97], aux = [257], level_config = [[[1 : i32, 1 : i32]]], alpha = 1, dnum = 0>

module attributes {cheddar_gpu.params = #params} {
  func.func @test_grid_truncation(%out: memref<1x64xi32>, %lhs: memref<1x64xi32>, %rhs: memref<1x64xi32>) {
    // expected-error @below {{total threads (64) must be positive and divisible by block dimension (256)}}
    // expected-error @below {{failed to legalize operation 'cheddar_gpu.add'}}
    cheddar_gpu.add %out, %lhs, %rhs <{
      lhs_form = #eval,
      rhs_form = #eval,
      out_form = #eval
    }> : memref<1x64xi32>
    return
  }
}
// -----

#eval = #cheddar_gpu.form<eval>
#params = #cheddar_gpu.params<word = 32, log_n = 31, ter = [17], main = [97], aux = [257], level_config = [[[1 : i32, 1 : i32]]], alpha = 1, dnum = 0>

module attributes {cheddar_gpu.params = #params} {
  func.func @test_overflow_int32(%out: memref<1x2147483648xi32>, %lhs: memref<1x2147483648xi32>, %rhs: memref<1x2147483648xi32>) {
    // expected-error @below {{elements or total threads exceed 32-bit signed integer range}}
    // expected-error @below {{failed to legalize operation 'cheddar_gpu.add'}}
    cheddar_gpu.add %out, %lhs, %rhs <{
      lhs_form = #eval,
      rhs_form = #eval,
      out_form = #eval
    }> : memref<1x2147483648xi32>
    return
  }
}

// -----

#eval = #cheddar_gpu.form<eval>
#params_add = #cheddar_gpu.params<word = 32, log_n = 16, ter = [17], main = [97, 193], aux = [257], level_config = [[[2 : i32, 1 : i32]]], alpha = 1, dnum = 0>

module attributes {cheddar_gpu.params = #params_add} {
  func.func @test_add_tensor_requires_buffer(%out: tensor<4x65536xi32>, %lhs: tensor<4x65536xi32>, %rhs: tensor<4x65536xi32>) -> tensor<4x65536xi32> {
    // expected-error @below {{requires buffer operands; run --one-shot-bufferize before --cheddar-gpu-to-gpu}}
    // expected-error @below {{failed to legalize operation 'cheddar_gpu.add'}}
    %res = cheddar_gpu.add %out, %lhs, %rhs <{
      lhs_form = #eval,
      rhs_form = #eval,
      out_form = #eval
    }> : tensor<4x65536xi32> -> tensor<4x65536xi32>
    return %res : tensor<4x65536xi32>
  }
}
