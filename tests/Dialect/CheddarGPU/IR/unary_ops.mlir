// RUN: heir-opt --verify-roundtrip %s | FileCheck %s

// CHECK: module attributes
// CHECK-SAME: cheddar_gpu.params = #cheddar_gpu.params<word = 32, log_n = 16, ter = [1000], main = [2000, 3000], aux = [4000], level_config = {{.*}}, alpha = 1, dnum = 0>
module attributes {
  cheddar_gpu.params = #cheddar_gpu.params<word = 32, log_n = 16, ter = [1000], main = [2000, 3000], aux = [4000], level_config = [[[2 : i32, 1 : i32]]], alpha = 1, dnum = 0>
} {

  // CHECK: @test_neg
  func.func @test_neg(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.neg %{{.*}}, %{{.*}} <{{.*}}in_form = eval{{.*}}> : memref<4x65536xi32>
    cheddar_gpu.neg %out, %in <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x65536xi32>
    return
  }

}
