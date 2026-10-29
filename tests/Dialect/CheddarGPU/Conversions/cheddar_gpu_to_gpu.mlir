// RUN: heir-opt %s --cheddar-gpu-to-gpu --gpu-kernel-outlining --mod-arith-to-arith | FileCheck %s

// CHECK-NOT: cheddar_gpu.
// CHECK-NOT: arith.remui
// CHECK-NOT: arith.remsi

module attributes {
  cheddar_gpu.params = #cheddar_gpu.params<word = 32, log_n = 16, ter = [30539777], main = [974258177, 1081212929], aux = [2147352577], level_config = [[[2 : i32, 1 : i32]]], alpha = 1, dnum = 0>
} {

  // CHECK: gpu.module @cheddar_kernels attributes {dlti.dl_spec = #dlti.dl_spec<
  // CHECK-SAME: "dlti.endianness" = "little"
  // CHECK-SAME: i64 = dense<64> : vector<2xi64>
  // CHECK-SAME: i128 = dense<128> : vector<2xi64>
  // CHECK-SAME: "dlti.legal_int_widths" = array<i32: 16, 32, 64>
  // CHECK-SAME: index = 64 : i64>

  // CHECK: func.func @test_add
  // CHECK-NOT: cheddar_gpu.
  // CHECK: gpu.launch_func
  func.func @test_add(%out: memref<4x65536xi32>, %lhs: memref<4x65536xi32>, %rhs: memref<4x65536xi32>) {
    cheddar_gpu.add %out, %lhs, %rhs <{
      lhs_form = #cheddar_gpu.form<eval>,
      rhs_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: func.func @test_add_limb_range
  // CHECK: gpu.launch_func
  func.func @test_add_limb_range(%out: memref<4x65536xi32>, %lhs: memref<4x65536xi32>, %rhs: memref<4x65536xi32>) {
    cheddar_gpu.add %out, %lhs, %rhs <{
      lhs_form = #cheddar_gpu.form<eval>,
      rhs_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>,
      lhs_limb_range = array<i64: 1, 2>,
      rhs_limb_range = array<i64: 1, 2>,
      out_limb_range = array<i64: 1, 2>
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: func.func @test_sub
  // CHECK: gpu.launch_func
  func.func @test_sub(%out: memref<4x65536xi32>, %lhs: memref<4x65536xi32>, %rhs: memref<4x65536xi32>) {
    cheddar_gpu.sub %out, %lhs, %rhs <{
      lhs_form = #cheddar_gpu.form<eval>,
      rhs_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: func.func @test_neg
  // CHECK: gpu.launch_func
  func.func @test_neg(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
    cheddar_gpu.neg %out, %in <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: func.func @test_add_const
  // CHECK: gpu.launch_func
  func.func @test_add_const(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>, %cst: memref<4xi32>) {
    cheddar_gpu.add_const %out, %in, %cst <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x65536xi32>, memref<4xi32>
    return
  }

  // CHECK: func.func @test_sub_const
  // CHECK: gpu.launch_func
  func.func @test_sub_const(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>, %cst: memref<4xi32>) {
    cheddar_gpu.sub_const %out, %in, %cst <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x65536xi32>, memref<4xi32>
    return
  }

  // CHECK: func.func @test_sub_opposite_const
  // CHECK: gpu.launch_func
  func.func @test_sub_opposite_const(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>, %cst: memref<4xi32>) {
    cheddar_gpu.sub_opposite_const %out, %in, %cst <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x65536xi32>, memref<4xi32>
    return
  }

  // CHECK: func.func @test_mul
  // CHECK: gpu.launch_func
  func.func @test_mul(%out: memref<4x65536xi32>, %lhs: memref<4x65536xi32>, %rhs: memref<4x65536xi32>) {
    cheddar_gpu.mul %out, %lhs, %rhs <{
      lhs_form = #cheddar_gpu.form<eval>,
      rhs_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: func.func @test_mul_const
  // CHECK: gpu.launch_func
  func.func @test_mul_const(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>, %cst: memref<4xi32>) {
    cheddar_gpu.mul_const %out, %in, %cst <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x65536xi32>, memref<4xi32>
    return
  }

  // CHECK: func.func @test_cp_accum
  // CHECK: gpu.launch_func
  func.func @test_cp_accum(%out: memref<4x65536xi32>, %c0: memref<4x65536xi32>, %c1: memref<4x65536xi32>, %p0: memref<4x65536xi32>, %p1: memref<4xi32>) {
    cheddar_gpu.cp_accum %out, ct_srcs(%c0, %c1), common_srcs(%p0, %p1) <{
      out_form = #cheddar_gpu.form<eval>
    }> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4xi32>) -> ()
    return
  }

  // CHECK: func.func @test_cp_accum_add
  // CHECK: gpu.launch_func
  func.func @test_cp_accum_add(%out: memref<4x65536xi32>, %acc: memref<4x65536xi32>, %c0: memref<4x65536xi32>, %p0: memref<4x65536xi32>) {
    cheddar_gpu.cp_accum_add %out, acc(%acc), ct_srcs(%c0), common_srcs(%p0) <{
      out_form = #cheddar_gpu.form<eval>
    }> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>) -> ()
    return
  }

  // CHECK: func.func @test_tensor
  // CHECK: gpu.launch_func
  func.func @test_tensor(%b: memref<4x65536xi32>, %a: memref<4x65536xi32>, %r: memref<4x65536xi32>,
                         %s1_b: memref<4x65536xi32>, %s1_a: memref<4x65536xi32>,
                         %s2_b: memref<4x65536xi32>, %s2_a: memref<4x65536xi32>) {
    cheddar_gpu.tensor %b, %a, %r, %s1_b, %s1_a, %s2_b, %s2_a <{
      lhs_form = #cheddar_gpu.form<eval>,
      rhs_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: func.func @test_tensor_square
  // CHECK: gpu.launch_func
  func.func @test_tensor_square(%b: memref<4x65536xi32>, %a: memref<4x65536xi32>, %r: memref<4x65536xi32>,
                                %s_b: memref<4x65536xi32>, %s_a: memref<4x65536xi32>) {
    cheddar_gpu.tensor_square %b, %a, %r, %s_b, %s_a <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: func.func @test_mul_imag_unit
  // CHECK: gpu.launch_func
  func.func @test_mul_imag_unit(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>, %i_unit: memref<4xi32>) {
    cheddar_gpu.mul_imag_unit %out, %in, %i_unit <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x65536xi32>, memref<4xi32>
    return
  }

  // CHECK: func.func @test_permute_galois
  // CHECK: gpu.launch_func
  func.func @test_permute_galois(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
    cheddar_gpu.permute %out, %in <{
      galois = 5 : i64,
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: func.func @test_permute_conj
  // CHECK: gpu.launch_func
  func.func @test_permute_conj(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
    cheddar_gpu.permute %out, %in <{
      conj,
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: func.func @test_permute_accum
  // CHECK: gpu.launch_func
  func.func @test_permute_accum(%out: memref<4x65536xi32>, %s0: memref<4x65536xi32>, %s1: memref<4x65536xi32>) {
    cheddar_gpu.permute_accum %out, srcs(%s0, %s1) <{
      galois_factors = array<i64: 5, 25>,
      in_forms = [#cheddar_gpu.form<eval>, #cheddar_gpu.form<eval>],
      out_form = #cheddar_gpu.form<eval>
    }> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>) -> ()
    return
  }

  // CHECK: func.func @test_permute_accum_with_acc
  // CHECK: gpu.launch_func
  func.func @test_permute_accum_with_acc(%out: memref<4x65536xi32>, %acc: memref<4x65536xi32>, %s0: memref<4x65536xi32>, %s1: memref<4x65536xi32>) {
    cheddar_gpu.permute_accum %out, acc(%acc), srcs(%s0, %s1) <{
      galois_factors = array<i64: 5, 25>,
      in_forms = [#cheddar_gpu.form<eval>, #cheddar_gpu.form<eval>],
      out_form = #cheddar_gpu.form<eval>
    }> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>) -> ()
    return
  }

  // CHECK: func.func @test_zero_pad
  // CHECK: gpu.launch_func
  func.func @test_zero_pad(%out: memref<4x65536xi32>) {
    cheddar_gpu.zero_pad %out <{limb_range = array<i64: 1, 3>}> : memref<4x65536xi32>
    return
  }

  // CHECK: func.func @test_mod_up_to_max
  // CHECK: gpu.launch_func
  func.func @test_mod_up_to_max(%out: memref<4x65536xi32>, %in: memref<1x65536xi32>) {
    cheddar_gpu.mod_up_to_max %out, %in <{
      in_form = #cheddar_gpu.form<coeff>,
      out_form = #cheddar_gpu.form<coeff>,
      num_aux = 1 : i64,
      prime_offset = 0 : i64
    }> : memref<4x65536xi32>, memref<1x65536xi32>
    return
  }

  // CHECK: func.func @test_cp_accum_extended
  // CHECK: gpu.launch_func
  func.func @test_cp_accum_extended(%out: memref<3x65536xi32>, %ct: memref<4x65536xi32>, %c: memref<3x65536xi32>) {
    cheddar_gpu.cp_accum %out, ct_srcs(%ct), common_srcs(%c) <{
      out_form = #cheddar_gpu.form<eval>,
      num_aux = 1 : i64,
      prime_offset = 0 : i64
    }> : (memref<3x65536xi32>, memref<4x65536xi32>, memref<3x65536xi32>) -> ()
    return
  }

  // CHECK: func.func @test_hoist_bs_fused
  // CHECK: gpu.launch_func {{.*}}@cheddar_kernels::@cheddar_gpu_hoist_bs_fused_w32_deg65536_k1_r2_L3
  func.func @test_hoist_bs_fused(
      %out_bx0: memref<3x65536xi32>, %out_bx1: memref<3x65536xi32>,
      %out_ax0: memref<3x65536xi32>, %out_ax1: memref<3x65536xi32>,
      %mod_up0: memref<3x65536xi32>, %evk_slab: memref<?xi8>,
      %in_bx_pseudo_modup: memref<2x65536xi32>) {
    cheddar_gpu.hoist_bs_fused
      out_bx(%out_bx0, %out_bx1),
      out_ax(%out_ax0, %out_ax1),
      mod_up(%mod_up0),
      %evk_slab, %in_bx_pseudo_modup <{
        evk_byte_offsets = array<i64: 0, 4096>,
        evk_num_q = array<i64: 2, 2>,
        galois_factors = array<i64: 5, 25>,
        num_aux = 1 : i64,
        num_q = 2 : i64,
        prime_offset = 0 : i64
      }> : (memref<3x65536xi32>, memref<3x65536xi32>,
            memref<3x65536xi32>, memref<3x65536xi32>,
            memref<3x65536xi32>, memref<?xi8>,
            memref<2x65536xi32>) -> ()
    return
  }

  // CHECK: func.func @test_hoist_gs_fused
  // CHECK: gpu.launch_func {{.*}}@cheddar_kernels::@cheddar_gpu_hoist_gs_fused_w32_deg65536_b2_g2_L3
  func.func @test_hoist_gs_fused(
      %out_bx0: memref<3x65536xi32>, %out_bx1: memref<3x65536xi32>,
      %out_ax0: memref<3x65536xi32>, %out_ax1: memref<3x65536xi32>,
      %in_bx0: memref<3x65536xi32>, %in_bx1: memref<3x65536xi32>,
      %in_ax0: memref<3x65536xi32>, %in_ax1: memref<3x65536xi32>,
      %consts_slab: memref<?xi8>) {
    cheddar_gpu.hoist_gs_fused
      out_bx(%out_bx0, %out_bx1),
      out_ax(%out_ax0, %out_ax1),
      in_bx(%in_bx0, %in_bx1),
      in_ax(%in_ax0, %in_ax1),
      %consts_slab <{
        num_aux = 1 : i64,
        num_q = 2 : i64,
        prime_offset = 0 : i64,
        pt_limbs = 3 : i64,
        pt_word_offsets = array<i64: 0, 131072, -1, 262144>
      }> : (memref<3x65536xi32>, memref<3x65536xi32>,
            memref<3x65536xi32>, memref<3x65536xi32>,
            memref<3x65536xi32>, memref<3x65536xi32>,
            memref<3x65536xi32>, memref<3x65536xi32>,
            memref<?xi8>) -> ()
    return
  }
}
