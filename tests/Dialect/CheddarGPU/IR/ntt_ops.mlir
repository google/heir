// RUN: heir-opt --verify-roundtrip %s | FileCheck %s

// CHECK: func.func @test_ntt_basic
func.func @test_ntt_basic(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
  // CHECK: cheddar_gpu.ntt %{{.*}}, %{{.*}} <montgomery_conversion = false> : (memref<4x65536xi32>, memref<4x65536xi32>) -> ()
  cheddar_gpu.ntt %out, %in <{
    montgomery_conversion = false
  }> : (memref<4x65536xi32>, memref<4x65536xi32>) -> ()
  return
}

// CHECK: func.func @test_ntt_mont_conv
func.func @test_ntt_mont_conv(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
  // CHECK: cheddar_gpu.ntt %{{.*}}, %{{.*}} <montgomery_conversion = true> : (memref<4x65536xi32>, memref<4x65536xi32>) -> ()
  cheddar_gpu.ntt %out, %in <{
    montgomery_conversion = true
  }> : (memref<4x65536xi32>, memref<4x65536xi32>) -> ()
  return
}

// CHECK: func.func @test_ntt_with_limb_range_and_skip_range
func.func @test_ntt_with_limb_range_and_skip_range(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
  // CHECK: cheddar_gpu.ntt %{{.*}}, %{{.*}} <{{.*}}skip_range = [1, 3]{{.*}}> : (memref<4x65536xi32>, memref<4x65536xi32>) -> ()
  cheddar_gpu.ntt %out, %in <{
    montgomery_conversion = false,
    in_limb_range = array<i64: 0, 4>,
    out_limb_range = array<i64: 0, 4>,
    skip_range = array<i64: 1, 3>
  }> : (memref<4x65536xi32>, memref<4x65536xi32>) -> ()
  return
}

// CHECK: func.func @test_ntt_with_skip_copy_src
func.func @test_ntt_with_skip_copy_src(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>, %skip: memref<4x65536xi32>) {
  // CHECK: cheddar_gpu.ntt %{{.*}}, %{{.*}}, skip_copy_src(%{{.*}}) <{{.*}}skip_range = [1, 3]{{.*}}> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>) -> ()
  cheddar_gpu.ntt %out, %in, skip_copy_src(%skip) <{
    montgomery_conversion = false,
    in_limb_range = array<i64: 0, 4>,
    out_limb_range = array<i64: 0, 4>,
    skip_range = array<i64: 1, 3>
  }> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>) -> ()
  return
}

// CHECK: func.func @test_intt_basic
func.func @test_intt_basic(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
  // CHECK: cheddar_gpu.intt %{{.*}}, %{{.*}} <{{.*}}epilogue = none{{.*}}> : (memref<4x65536xi32>, memref<4x65536xi32>) -> ()
  cheddar_gpu.intt %out, %in <{
    montgomery_conversion = true,
    epilogue = #cheddar_gpu.intt_epilogue<none>
  }> : (memref<4x65536xi32>, memref<4x65536xi32>) -> ()
  return
}

// CHECK: func.func @test_intt_mont_false
func.func @test_intt_mont_false(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
  // CHECK: cheddar_gpu.intt %{{.*}}, %{{.*}} <{{.*}}epilogue = none{{.*}}> : (memref<4x65536xi32>, memref<4x65536xi32>) -> ()
  cheddar_gpu.intt %out, %in <{
    montgomery_conversion = false,
    epilogue = #cheddar_gpu.intt_epilogue<none>
  }> : (memref<4x65536xi32>, memref<4x65536xi32>) -> ()
  return
}

// CHECK: func.func @test_intt_mult_const
func.func @test_intt_mult_const(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>, %cst: memref<4xi32>) {
  // CHECK: cheddar_gpu.intt %{{.*}}, %{{.*}}, cst(%{{.*}}) <{{.*}}epilogue = mult_const{{.*}}> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4xi32>) -> ()
  cheddar_gpu.intt %out, %in, cst(%cst) <{
    montgomery_conversion = true,
    epilogue = #cheddar_gpu.intt_epilogue<mult_const>
  }> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4xi32>) -> ()
  return
}

// CHECK: func.func @test_intt_mult_const_normalize
func.func @test_intt_mult_const_normalize(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>, %cst: memref<4xi32>) {
  // CHECK: cheddar_gpu.intt %{{.*}}, %{{.*}}, cst(%{{.*}}) <{{.*}}epilogue = mult_const_normalize{{.*}}> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4xi32>) -> ()
  cheddar_gpu.intt %out, %in, cst(%cst) <{
    montgomery_conversion = true,
    epilogue = #cheddar_gpu.intt_epilogue<mult_const_normalize>
  }> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4xi32>) -> ()
  return
}
