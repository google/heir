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
#params = #cheddar_gpu.params<word = 32, log_n = 8, ter = [17], main = [97], aux = [257], level_config = [[[1 : i32, 1 : i32]]], alpha = 1, dnum = 0>

module attributes {cheddar_gpu.params = #params} {
  func.func @test_degree_mismatch(%out: memref<1x128xi32>, %lhs: memref<1x128xi32>, %rhs: memref<1x128xi32>) {
    // expected-error @below {{operand degree (128) does not match module params degree (256)}}
    // expected-error @below {{failed to legalize operation 'cheddar_gpu.add'}}
    cheddar_gpu.add %out, %lhs, %rhs <{
      lhs_form = #eval,
      rhs_form = #eval,
      out_form = #eval
    }> : memref<1x128xi32>
    return
  }
}

// -----

#eval = #cheddar_gpu.form<eval>
#params = #cheddar_gpu.params<word = 32, log_n = 8, ter = [17], main = [97], aux = [257], level_config = [[[1 : i32, 1 : i32]]], alpha = 1, dnum = 0>

module attributes {cheddar_gpu.params = #params} {
  func.func @test_word_mismatch(%out: memref<1x256xi64>, %lhs: memref<1x256xi64>, %rhs: memref<1x256xi64>) {
    // expected-error @below {{operand bitwidth (64) does not match module params word (32)}}
    // expected-error @below {{failed to legalize operation 'cheddar_gpu.add'}}
    cheddar_gpu.add %out, %lhs, %rhs <{
      lhs_form = #eval,
      rhs_form = #eval,
      out_form = #eval
    }> : memref<1x256xi64>
    return
  }
}

// -----

#eval = #cheddar_gpu.form<eval>
#params = #cheddar_gpu.params<word = 32, log_n = 8, ter = [17], main = [97], aux = [257], level_config = [[[1 : i32, 1 : i32]]], alpha = 1, dnum = 0>

module attributes {cheddar_gpu.params = #params} {
  func.func @test_num_aux_exceeds_alpha(%out: memref<2x256xi32>, %lhs: memref<2x256xi32>, %rhs: memref<2x256xi32>) {
    // expected-error @below {{num_aux (2) exceeds module params aux limb count (1)}}
    // expected-error @below {{failed to legalize operation 'cheddar_gpu.add'}}
    cheddar_gpu.add %out, %lhs, %rhs <{
      lhs_form = #eval,
      rhs_form = #eval,
      out_form = #eval,
      num_aux = 2 : i64
    }> : memref<2x256xi32>
    return
  }
}

// -----

#eval = #cheddar_gpu.form<eval>
#params = #cheddar_gpu.params<word = 32, log_n = 8, ter = [17], main = [97], aux = [257], level_config = [[[1 : i32, 1 : i32]]], alpha = 1, dnum = 0>

module attributes {cheddar_gpu.params = #params} {
  func.func @test_num_q_exceeds_ter_plus_main(%out: memref<3x256xi32>, %lhs: memref<3x256xi32>, %rhs: memref<3x256xi32>) {
    // expected-error @below {{main limb count (3) exceeds module params ter + main limb count (2)}}
    // expected-error @below {{failed to legalize operation 'cheddar_gpu.add'}}
    cheddar_gpu.add %out, %lhs, %rhs <{
      lhs_form = #eval,
      rhs_form = #eval,
      out_form = #eval
    }> : memref<3x256xi32>
    return
  }
}
