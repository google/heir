// RUN: heir-opt --verify-roundtrip %s | FileCheck %s

// CHECK: module attributes
// CHECK-SAME: cheddar_gpu.params = #cheddar_gpu.params<word = 32, log_n = 16, ter = [1000], main = [2000, 3000], aux = [4000], level_config = {{.*}}, alpha = 1, dnum = 0>
module attributes {
  cheddar_gpu.params = #cheddar_gpu.params<word = 32, log_n = 16, ter = [1000], main = [2000, 3000], aux = [4000], level_config = [[[2 : i32, 1 : i32]]], alpha = 1, dnum = 0>
} {

  // CHECK: @test_add_coeff
  func.func @test_add_coeff(%out: memref<4x65536xi32>, %lhs: memref<4x65536xi32>, %rhs: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.add %{{.*}}, %{{.*}}, %{{.*}} <{{.*}}lhs_form = coeff{{.*}}> : memref<4x65536xi32>
    cheddar_gpu.add %out, %lhs, %rhs <{
      lhs_form = #cheddar_gpu.form<coeff>,
      rhs_form = #cheddar_gpu.form<coeff>,
      out_form = #cheddar_gpu.form<coeff>
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: @test_add_eval_with_limb_range
  func.func @test_add_eval_with_limb_range(%out: memref<4x65536xi32>, %lhs: memref<4x65536xi32>, %rhs: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.add %{{.*}}, %{{.*}}, %{{.*}} <{{.*}}lhs_form = eval{{.*}}lhs_limb_range = [0, 2]{{.*}}> : memref<4x65536xi32>
    cheddar_gpu.add %out, %lhs, %rhs <{
      lhs_form = #cheddar_gpu.form<eval>,
      rhs_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>,
      lhs_limb_range = array<i64: 0, 2>,
      rhs_limb_range = array<i64: 0, 2>,
      out_limb_range = array<i64: 0, 2>
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: @test_sub
  func.func @test_sub(%out: memref<4x65536xi32>, %lhs: memref<4x65536xi32>, %rhs: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.sub %{{.*}}, %{{.*}}, %{{.*}} <{{.*}}lhs_form = eval{{.*}}> : memref<4x65536xi32>
    cheddar_gpu.sub %out, %lhs, %rhs <{
      lhs_form = #cheddar_gpu.form<eval>,
      rhs_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: @test_mul
  func.func @test_mul(%out: memref<4x65536xi32>, %lhs: memref<4x65536xi32>, %rhs: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.mul %{{.*}}, %{{.*}}, %{{.*}} <{{.*}}lhs_form = eval{{.*}}> : memref<4x65536xi32>
    cheddar_gpu.mul %out, %lhs, %rhs <{
      lhs_form = #cheddar_gpu.form<eval>,
      rhs_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x65536xi32>
    return
  }

}
