// RUN: heir-opt --verify-roundtrip %s | FileCheck %s

// CHECK: module attributes
// CHECK-SAME: cheddar_gpu.params = #cheddar_gpu.params<word = 32, log_n = 16, ter = [1000], main = [2000, 3000], aux = [4000], level_config = {{.*}}, alpha = 1, dnum = 0>
module attributes {
  cheddar_gpu.params = #cheddar_gpu.params<word = 32, log_n = 16, ter = [1000], main = [2000, 3000], aux = [4000], level_config = [[[2 : i32, 1 : i32]]], alpha = 1, dnum = 0>
} {

  // CHECK: @test_add_const
  func.func @test_add_const(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>, %cst: memref<4xi32>) {
    // CHECK: cheddar_gpu.add_const %{{.*}}, %{{.*}}, %{{.*}} <{{.*}}in_form = eval{{.*}}> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4xi32>) -> ()
    cheddar_gpu.add_const %out, %in, %cst <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4xi32>) -> ()
    return
  }

  // CHECK: @test_sub_const
  func.func @test_sub_const(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>, %cst: memref<4xi32>) {
    // CHECK: cheddar_gpu.sub_const %{{.*}}, %{{.*}}, %{{.*}} <{{.*}}in_form = eval{{.*}}> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4xi32>) -> ()
    cheddar_gpu.sub_const %out, %in, %cst <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4xi32>) -> ()
    return
  }

  // CHECK: @test_sub_opposite_const
  func.func @test_sub_opposite_const(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>, %cst: memref<4xi32>) {
    // CHECK: cheddar_gpu.sub_opposite_const %{{.*}}, %{{.*}}, %{{.*}} <{{.*}}in_form = eval{{.*}}> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4xi32>) -> ()
    cheddar_gpu.sub_opposite_const %out, %in, %cst <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4xi32>) -> ()
    return
  }

  // CHECK: @test_mul_const
  func.func @test_mul_const(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>, %cst: memref<4xi32>) {
    // CHECK: cheddar_gpu.mul_const %{{.*}}, %{{.*}}, %{{.*}} : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4xi32>) -> ()
    cheddar_gpu.mul_const %out, %in, %cst : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4xi32>) -> ()
    return
  }

  // CHECK: @test_mul_imag_unit
  func.func @test_mul_imag_unit(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>, %i_unit: memref<4xi32>) {
    // CHECK: cheddar_gpu.mul_imag_unit %{{.*}}, %{{.*}}, %{{.*}} : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4xi32>) -> ()
    cheddar_gpu.mul_imag_unit %out, %in, %i_unit : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4xi32>) -> ()
    return
  }

}
