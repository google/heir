// RUN: heir-opt --verify-roundtrip %s | FileCheck %s

// CHECK: module attributes
// CHECK-SAME: cheddar_gpu.params = #cheddar_gpu.params<word = 32, log_n = 16, ter = [1000], main = [2000, 3000], aux = [4000], level_config = {{.*}}, alpha = 1, dnum = 0>
module attributes {
  cheddar_gpu.params = #cheddar_gpu.params<word = 32, log_n = 16, ter = [1000], main = [2000, 3000], aux = [4000], level_config = [[[2 : i32, 1 : i32]]], alpha = 1, dnum = 0>
} {

  // CHECK: @test_copy
  func.func @test_copy(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.copy %{{.*}}, %{{.*}} : (memref<4x65536xi32>, memref<4x65536xi32>) -> ()
    cheddar_gpu.copy %out, %in : (memref<4x65536xi32>, memref<4x65536xi32>) -> ()
    return
  }

  // CHECK: @test_copy_with_ranges
  func.func @test_copy_with_ranges(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.copy %{{.*}}, %{{.*}} <{{.*}}in_limb_range = [0, 2]{{.*}}out_limb_range = [1, 2]{{.*}}> : (memref<4x65536xi32>, memref<4x65536xi32>) -> ()
    cheddar_gpu.copy %out, %in <{
      in_limb_range = array<i64: 0, 2>,
      out_limb_range = array<i64: 1, 2>
    }> : (memref<4x65536xi32>, memref<4x65536xi32>) -> ()
    return
  }

}
