// RUN: heir-opt --verify-roundtrip %s | FileCheck %s

// CHECK: module attributes
// CHECK-SAME: cheddar_gpu.params = #cheddar_gpu.params<word = 32, log_n = 16, ter = [1000], main = [2000, 3000], aux = [4000], level_config = {{.*}}, alpha = 1, dnum = 0>
module attributes {
  cheddar_gpu.params = #cheddar_gpu.params<word = 32, log_n = 16, ter = [1000], main = [2000, 3000], aux = [4000], level_config = [[[2 : i32, 1 : i32]]], alpha = 1, dnum = 0>
} {

  // CHECK: @test_cp_accum
  func.func @test_cp_accum(%out: memref<4x65536xi32>, %c0: memref<4x65536xi32>, %c1: memref<4x65536xi32>, %p0: memref<4x65536xi32>, %p1: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.cp_accum %{{.*}}, ct_srcs(%{{.*}}, %{{.*}}), common_srcs(%{{.*}}, %{{.*}}){{.*}} : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>) -> ()
    cheddar_gpu.cp_accum %out, ct_srcs(%c0, %c1), common_srcs(%p0, %p1) : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>) -> ()
    return
  }

  // CHECK: @test_cp_accum_regression_limb_range
  func.func @test_cp_accum_regression_limb_range(%out: memref<4x65536xi32>, %c0: memref<2x65536xi32>, %p0: memref<2xi32>) {
    // CHECK: cheddar_gpu.cp_accum %{{.*}}, ct_srcs(%{{.*}}), common_srcs(%{{.*}}) <{{.*}}out_limb_range = [2, 2]{{.*}}> : (memref<4x65536xi32>, memref<2x65536xi32>, memref<2xi32>) -> ()
    cheddar_gpu.cp_accum %out, ct_srcs(%c0), common_srcs(%p0) <{
      out_limb_range = array<i64: 2, 2>
    }> : (memref<4x65536xi32>, memref<2x65536xi32>, memref<2xi32>) -> ()
    return
  }

  // CHECK: @test_cp_accum_add
  func.func @test_cp_accum_add(%out: memref<4x65536xi32>, %acc: memref<4x65536xi32>, %c0: memref<4x65536xi32>, %p0: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.cp_accum_add %{{.*}}, acc(%{{.*}}), ct_srcs(%{{.*}}), common_srcs(%{{.*}}){{.*}} : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>) -> ()
    cheddar_gpu.cp_accum_add %out, acc(%acc), ct_srcs(%c0), common_srcs(%p0) : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>) -> ()
    return
  }

  // CHECK: @test_cp_accum_2
  func.func @test_cp_accum_2(%out_b: memref<4x65536xi32>, %out_a: memref<4x65536xi32>,
                             %cb0: memref<4x65536xi32>, %ca0: memref<4x65536xi32>,
                             %p0: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.cp_accum_2 %{{.*}}, %{{.*}}, ct_bx_srcs(%{{.*}}), ct_ax_srcs(%{{.*}}), common_srcs(%{{.*}}){{.*}} : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>) -> ()
    cheddar_gpu.cp_accum_2 %out_b, %out_a, ct_bx_srcs(%cb0), ct_ax_srcs(%ca0), common_srcs(%p0) : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>) -> ()
    return
  }

  // CHECK: @test_cp_accum_add_2
  func.func @test_cp_accum_add_2(%out_b: memref<4x65536xi32>, %out_a: memref<4x65536xi32>,
                                 %acc_b: memref<4x65536xi32>, %acc_a: memref<4x65536xi32>,
                                 %cb0: memref<4x65536xi32>, %ca0: memref<4x65536xi32>,
                                 %p0: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.cp_accum_add_2 %{{.*}}, %{{.*}}, acc(%{{.*}}, %{{.*}}), ct_bx_srcs(%{{.*}}), ct_ax_srcs(%{{.*}}), common_srcs(%{{.*}}){{.*}} : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>) -> ()
    cheddar_gpu.cp_accum_add_2 %out_b, %out_a, acc(%acc_b, %acc_a), ct_bx_srcs(%cb0), ct_ax_srcs(%ca0), common_srcs(%p0) : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>) -> ()
    return
  }

  // CHECK: @test_permute_galois
  func.func @test_permute_galois(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.permute %{{.*}}, %{{.*}} <{{.*}}galois = 5{{.*}}> : memref<4x65536xi32>
    cheddar_gpu.permute %out, %in <{
      galois = 5 : i64
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: @test_permute_conj
  func.func @test_permute_conj(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.permute %{{.*}}, %{{.*}} <{{.*}}conj{{.*}}> : memref<4x65536xi32>
    cheddar_gpu.permute %out, %in <{
      conj
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: @test_permute_2
  func.func @test_permute_2(%out_b: memref<4x65536xi32>, %out_a: memref<4x65536xi32>,
                            %in_b: memref<4x65536xi32>, %in_a: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.permute_2 ins(%{{.*}}, %{{.*}}) outs(%{{.*}}, %{{.*}}) <{{.*}}galois = 5{{.*}}> : memref<4x65536xi32>
    cheddar_gpu.permute_2 ins(%in_b, %in_a)
                          outs(%out_b, %out_a) <{
      galois = 5 : i64
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: @test_permute_accum
  func.func @test_permute_accum(%out: memref<4x65536xi32>, %s0: memref<4x65536xi32>, %s1: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.permute_accum %{{.*}}, srcs(%{{.*}}, %{{.*}}) <{{.*}}galois_factors = [5, 25]{{.*}}> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>) -> ()
    cheddar_gpu.permute_accum %out, srcs(%s0, %s1) <{
      galois_factors = array<i64: 5, 25>
    }> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>) -> ()
    return
  }

  // CHECK: @test_permute_accum_2
  func.func @test_permute_accum_2(%out_b: memref<4x65536xi32>, %out_a: memref<4x65536xi32>,
                                  %sb0: memref<4x65536xi32>, %sa0: memref<4x65536xi32>,
                                  %sb1: memref<4x65536xi32>, %sa1: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.permute_accum_2 %{{.*}}, %{{.*}}, srcs_bx(%{{.*}}, %{{.*}}), srcs_ax(%{{.*}}, %{{.*}}) <{{.*}}galois_factors = [5, 25]{{.*}}> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>) -> ()
    cheddar_gpu.permute_accum_2 %out_b, %out_a, srcs_bx(%sb0, %sb1), srcs_ax(%sa0, %sa1) <{
      galois_factors = array<i64: 5, 25>
    }> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>) -> ()
    return
  }

}
