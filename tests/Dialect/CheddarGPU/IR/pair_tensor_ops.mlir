// RUN: heir-opt --verify-roundtrip %s | FileCheck %s

// CHECK: module attributes
// CHECK-SAME: cheddar_gpu.params = #cheddar_gpu.params<word = 32, log_n = 16, ter = [1000], main = [2000, 3000], aux = [4000], level_config = {{.*}}, alpha = 1, dnum = 0>
module attributes {
  cheddar_gpu.params = #cheddar_gpu.params<word = 32, log_n = 16, ter = [1000], main = [2000, 3000], aux = [4000], level_config = [[[2 : i32, 1 : i32]]], alpha = 1, dnum = 0>
} {

  // CHECK: @test_tensor
  func.func @test_tensor(%b: memref<4x65536xi32>, %a: memref<4x65536xi32>, %r: memref<4x65536xi32>,
                         %s1_b: memref<4x65536xi32>, %s1_a: memref<4x65536xi32>,
                         %s2_b: memref<4x65536xi32>, %s2_a: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.tensor ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) outs(%{{.*}}, %{{.*}}, %{{.*}}) : memref<4x65536xi32>
    cheddar_gpu.tensor ins(%s1_b, %s1_a, %s2_b, %s2_a)
                       outs(%b, %a, %r) : memref<4x65536xi32>
    return
  }

  // CHECK: @test_tensor_with_limb_range
  func.func @test_tensor_with_limb_range(%b: memref<4x65536xi32>, %a: memref<4x65536xi32>, %r: memref<4x65536xi32>,
                                         %s1_b: memref<4x65536xi32>, %s1_a: memref<4x65536xi32>,
                                         %s2_b: memref<4x65536xi32>, %s2_a: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.tensor ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) outs(%{{.*}}, %{{.*}}, %{{.*}}) <out_limb_range = [0, 2]> : memref<4x65536xi32>
    cheddar_gpu.tensor ins(%s1_b, %s1_a, %s2_b, %s2_a)
                       outs(%b, %a, %r) <{
      out_limb_range = array<i64: 0, 2>
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: @test_tensor_square
  func.func @test_tensor_square(%b: memref<4x65536xi32>, %a: memref<4x65536xi32>, %r: memref<4x65536xi32>,
                                %s_b: memref<4x65536xi32>, %s_a: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.tensor_square ins(%{{.*}}, %{{.*}}) outs(%{{.*}}, %{{.*}}, %{{.*}}) : memref<4x65536xi32>
    cheddar_gpu.tensor_square ins(%s_b, %s_a)
                              outs(%b, %a, %r) : memref<4x65536xi32>
    return
  }

  // CHECK: @test_add_2
  func.func @test_add_2(%out_b: memref<4x65536xi32>, %out_a: memref<4x65536xi32>,
                        %lhs_b: memref<4x65536xi32>, %lhs_a: memref<4x65536xi32>,
                        %rhs_b: memref<4x65536xi32>, %rhs_a: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.add_2 ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) outs(%{{.*}}, %{{.*}}) <{{.*}}lhs_form = coeff{{.*}}> : memref<4x65536xi32>
    cheddar_gpu.add_2 ins(%lhs_b, %lhs_a, %rhs_b, %rhs_a)
                      outs(%out_b, %out_a) <{
      lhs_form = #cheddar_gpu.form<coeff>,
      rhs_form = #cheddar_gpu.form<coeff>,
      out_form = #cheddar_gpu.form<coeff>
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: @test_add_2_with_limb_range
  func.func @test_add_2_with_limb_range(%out_b: memref<4x65536xi32>, %out_a: memref<4x65536xi32>,
                                        %lhs_b: memref<4x65536xi32>, %lhs_a: memref<4x65536xi32>,
                                        %rhs_b: memref<4x65536xi32>, %rhs_a: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.add_2 ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) outs(%{{.*}}, %{{.*}}) <{{.*}}lhs_limb_range = [0, 2]{{.*}}> : memref<4x65536xi32>
    cheddar_gpu.add_2 ins(%lhs_b, %lhs_a, %rhs_b, %rhs_a)
                      outs(%out_b, %out_a) <{
      lhs_form = #cheddar_gpu.form<eval>,
      rhs_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>,
      lhs_limb_range = array<i64: 0, 2>,
      rhs_limb_range = array<i64: 0, 2>,
      out_limb_range = array<i64: 0, 2>,
      num_aux = 1 : i64
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: @test_sub_2
  func.func @test_sub_2(%out_b: memref<4x65536xi32>, %out_a: memref<4x65536xi32>,
                        %lhs_b: memref<4x65536xi32>, %lhs_a: memref<4x65536xi32>,
                        %rhs_b: memref<4x65536xi32>, %rhs_a: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.sub_2 ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) outs(%{{.*}}, %{{.*}}) <{{.*}}lhs_form = eval{{.*}}> : memref<4x65536xi32>
    cheddar_gpu.sub_2 ins(%lhs_b, %lhs_a, %rhs_b, %rhs_a)
                      outs(%out_b, %out_a) <{
      lhs_form = #cheddar_gpu.form<eval>,
      rhs_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: @test_neg_2
  func.func @test_neg_2(%out_b: memref<4x65536xi32>, %out_a: memref<4x65536xi32>,
                        %in_b: memref<4x65536xi32>, %in_a: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.neg_2 ins(%{{.*}}, %{{.*}}) outs(%{{.*}}, %{{.*}}) <{{.*}}in_form = eval{{.*}}> : memref<4x65536xi32>
    cheddar_gpu.neg_2 ins(%in_b, %in_a)
                      outs(%out_b, %out_a) <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x65536xi32>
    return
  }

}
