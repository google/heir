// RUN: heir-opt --cheddar-gpu-to-scf %s | FileCheck %s

module attributes {
  cheddar_gpu.params = #cheddar_gpu.params<word = 32, log_n = 4, ter = [1000000007], main = [1000000009], aux = [1000000021], level_config = [[[1 : i32, 1 : i32]]], alpha = 1, dnum = 0>
} {
  // CHECK: @test_add
  // CHECK-NOT: cheddar_gpu.add
  // CHECK-NOT: arith.remui
  // CHECK-NOT: arith.remsi
  // CHECK: scf.for
  // CHECK: mod_arith.add
  func.func @test_add(%out: memref<3x16xi32>, %lhs: memref<3x16xi32>, %rhs: memref<3x16xi32>) {
    cheddar_gpu.add %out, %lhs, %rhs <{
      lhs_form = #cheddar_gpu.form<eval>,
      rhs_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<3x16xi32>
    return
  }

  // CHECK: @test_sub
  // CHECK-NOT: cheddar_gpu.sub
  // CHECK-NOT: arith.remui
  // CHECK-NOT: arith.remsi
  // CHECK: scf.for
  // CHECK: mod_arith.sub
  func.func @test_sub(%out: memref<3x16xi32>, %lhs: memref<3x16xi32>, %rhs: memref<3x16xi32>) {
    cheddar_gpu.sub %out, %lhs, %rhs <{
      lhs_form = #cheddar_gpu.form<eval>,
      rhs_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<3x16xi32>
    return
  }

  // CHECK: @test_neg
  // CHECK-NOT: cheddar_gpu.neg
  // CHECK-NOT: arith.remui
  // CHECK-NOT: arith.remsi
  // CHECK: scf.for
  // CHECK: mod_arith.sub
  func.func @test_neg(%out: memref<3x16xi32>, %in: memref<3x16xi32>) {
    cheddar_gpu.neg %out, %in <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<3x16xi32>
    return
  }

  // CHECK: @test_add_const
  // CHECK-NOT: cheddar_gpu.add_const
  // CHECK: scf.for
  // CHECK: mod_arith.add
  func.func @test_add_const(%out: memref<3x16xi32>, %in: memref<3x16xi32>, %cst: memref<3xi32>) {
    cheddar_gpu.add_const %out, %in, %cst <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<3x16xi32>, memref<3xi32>
    return
  }

  // CHECK: @test_sub_const
  // CHECK-NOT: cheddar_gpu.sub_const
  // CHECK: scf.for
  // CHECK: mod_arith.sub
  func.func @test_sub_const(%out: memref<3x16xi32>, %in: memref<3x16xi32>, %cst: memref<3xi32>) {
    cheddar_gpu.sub_const %out, %in, %cst <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<3x16xi32>, memref<3xi32>
    return
  }

  // CHECK: @test_sub_opposite_const
  // CHECK-NOT: cheddar_gpu.sub_opposite_const
  // CHECK: scf.for
  // CHECK: mod_arith.sub
  func.func @test_sub_opposite_const(%out: memref<3x16xi32>, %in: memref<3x16xi32>, %cst: memref<3xi32>) {
    cheddar_gpu.sub_opposite_const %out, %in, %cst <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<3x16xi32>, memref<3xi32>
    return
  }

  // CHECK: @test_mul
  // CHECK-NOT: cheddar_gpu.mul
  // CHECK: scf.for
  // CHECK: mod_arith.mont_mul
  func.func @test_mul(%out: memref<3x16xi32>, %lhs: memref<3x16xi32>, %rhs: memref<3x16xi32>) {
    cheddar_gpu.mul %out, %lhs, %rhs <{
      lhs_form = #cheddar_gpu.form<eval>,
      rhs_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<3x16xi32>
    return
  }

  // CHECK: @test_mul_const
  // CHECK-NOT: cheddar_gpu.mul_const
  // CHECK: scf.for
  // CHECK: mod_arith.mont_mul
  func.func @test_mul_const(%out: memref<3x16xi32>, %in: memref<3x16xi32>, %cst: memref<3xi32>) {
    cheddar_gpu.mul_const %out, %in, %cst <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<3x16xi32>, memref<3xi32>
    return
  }

  // CHECK: @test_cp_accum
  // CHECK-NOT: cheddar_gpu.cp_accum
  // CHECK: scf.for
  // CHECK: mod_arith.mont_mul
  // CHECK: mod_arith.add
  func.func @test_cp_accum(%out: memref<3x16xi32>, %c1: memref<3x16xi32>, %c2: memref<3x16xi32>, %k1: memref<3xi32>, %k2: memref<3xi32>) {
    cheddar_gpu.cp_accum %out, ct_srcs(%c1, %c2), common_srcs(%k1, %k2) <{
      out_form = #cheddar_gpu.form<eval>
    }> : (memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>, memref<3xi32>, memref<3xi32>) -> ()
    return
  }

  // CHECK: @test_cp_accum_add
  // CHECK-NOT: cheddar_gpu.cp_accum_add
  // CHECK: scf.for
  // CHECK: mod_arith.mont_mul
  // CHECK: mod_arith.add
  func.func @test_cp_accum_add(%out: memref<3x16xi32>, %acc: memref<3x16xi32>, %c1: memref<3x16xi32>, %k1: memref<3xi32>) {
    cheddar_gpu.cp_accum_add %out, acc(%acc), ct_srcs(%c1), common_srcs(%k1) <{
      out_form = #cheddar_gpu.form<eval>
    }> : (memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>, memref<3xi32>) -> ()
    return
  }

  // CHECK: @test_tensor
  // CHECK-NOT: cheddar_gpu.tensor
  // CHECK: scf.for
  // CHECK: mod_arith.mont_mul
  func.func @test_tensor(%obx: memref<3x16xi32>, %oax: memref<3x16xi32>, %orx: memref<3x16xi32>,
                         %s1bx: memref<3x16xi32>, %s1ax: memref<3x16xi32>,
                         %s2bx: memref<3x16xi32>, %s2ax: memref<3x16xi32>) {
    cheddar_gpu.tensor %obx, %oax, %orx, %s1bx, %s1ax, %s2bx, %s2ax <{
      lhs_form = #cheddar_gpu.form<eval>,
      rhs_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<3x16xi32>
    return
  }

  // CHECK: @test_tensor_square
  // CHECK-NOT: cheddar_gpu.tensor_square
  // CHECK: scf.for
  // CHECK: mod_arith.mont_mul
  func.func @test_tensor_square(%obx: memref<3x16xi32>, %oax: memref<3x16xi32>, %orx: memref<3x16xi32>,
                                %sbx: memref<3x16xi32>, %sax: memref<3x16xi32>) {
    cheddar_gpu.tensor_square %obx, %oax, %orx, %sbx, %sax <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<3x16xi32>
    return
  }

  // CHECK: @test_mul_imag_unit
  // CHECK-NOT: cheddar_gpu.mul_imag_unit
  // CHECK: scf.for
  // CHECK: mod_arith.mont_mul
  func.func @test_mul_imag_unit(%out: memref<3x16xi32>, %in: memref<3x16xi32>, %iunit: memref<3xi32>) {
    cheddar_gpu.mul_imag_unit %out, %in, %iunit <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<3x16xi32>, memref<3xi32>
    return
  }

  // CHECK: @test_permute
  // CHECK-NOT: cheddar_gpu.permute
  // CHECK: scf.for
  // CHECK: memref.load
  // CHECK: memref.store
  func.func @test_permute(%out: memref<3x16xi32>, %in: memref<3x16xi32>) {
    cheddar_gpu.permute %out, %in <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>,
      galois = 5 : i64
    }> : memref<3x16xi32>
    return
  }

  // CHECK: @test_permute_accum
  // CHECK-NOT: cheddar_gpu.permute_accum
  // CHECK: scf.for
  // CHECK: mod_arith.add
  func.func @test_permute_accum(%out: memref<3x16xi32>, %acc: memref<3x16xi32>, %s1: memref<3x16xi32>, %s2: memref<3x16xi32>) {
    cheddar_gpu.permute_accum %out, acc(%acc), srcs(%s1, %s2) <{
      out_form = #cheddar_gpu.form<eval>,
      galois_factors = array<i64: 5, 25>
    }> : (memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>) -> ()
    return
  }

  // CHECK: @test_mod_down_epilogue_unfused
  // CHECK-NOT: cheddar_gpu.mod_down_epilogue
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: mod_arith.sub
  // CHECK: mod_arith.mont_mul
  func.func @test_mod_down_epilogue_unfused(%out: memref<3x16xi32>, %ntt_bconv: memref<3x16xi32>, %inv_p_prod: memref<3xi32>) {
    cheddar_gpu.mod_down_epilogue %out, %ntt_bconv, %inv_p_prod <pad_start = 0, pad_end = 3, in_form = eval, out_form = eval> : memref<3x16xi32>, memref<3x16xi32>, memref<3xi32>
    return
  }

  // CHECK: @test_mod_down_epilogue_fused
  // CHECK-NOT: cheddar_gpu.mod_down_epilogue
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: mod_arith.mont_mul
  // CHECK: mod_arith.sub
  // CHECK: mod_arith.mont_mul
  func.func @test_mod_down_epilogue_fused(%out: memref<3x16xi32>, %ntt_bconv: memref<3x16xi32>, %inv_p_prod: memref<3xi32>, %src2: memref<3x16xi32>, %src2_padding: memref<3xi32>) {
    cheddar_gpu.mod_down_epilogue %out, %ntt_bconv, %inv_p_prod, %src2, %src2_padding <pad_start = 0, pad_end = 3, in_form = eval, out_form = eval> : memref<3x16xi32>, memref<3x16xi32>, memref<3xi32>, memref<3x16xi32>, memref<3xi32>
    return
  }

  // CHECK: func.func @test_hoist_bs_fused
  // CHECK-NOT: cheddar_gpu.hoist_bs_fused
  // CHECK: scf.for
  // CHECK: mod_arith.mont_mul
  // CHECK: mod_arith.add
  func.func @test_hoist_bs_fused(
      %out_bx0: memref<2x16xi32>, %out_bx1: memref<2x16xi32>,
      %out_ax0: memref<2x16xi32>, %out_ax1: memref<2x16xi32>,
      %mod_up0: memref<2x16xi32>, %evk_slab: memref<?xi8>,
      %in_bx_pseudo_modup: memref<1x16xi32>) {
    cheddar_gpu.hoist_bs_fused
      out_bx(%out_bx0, %out_bx1),
      out_ax(%out_ax0, %out_ax1),
      mod_up(%mod_up0),
      %evk_slab, %in_bx_pseudo_modup <{
        evk_byte_offsets = array<i64: 0, 4096>,
        evk_num_q = array<i64: 1, 1>,
        galois_factors = array<i64: 5, 7>,
        num_aux = 1 : i64,
        num_q = 1 : i64,
        prime_offset = 0 : i64
      }> : (memref<2x16xi32>, memref<2x16xi32>,
            memref<2x16xi32>, memref<2x16xi32>,
            memref<2x16xi32>, memref<?xi8>,
            memref<1x16xi32>) -> ()
    return
  }

  // CHECK: func.func @test_hoist_gs_fused
  // CHECK-NOT: cheddar_gpu.hoist_gs_fused
  // CHECK: scf.for
  // CHECK: mod_arith.mont_mul
  // CHECK: mod_arith.add
  func.func @test_hoist_gs_fused(
      %out_bx0: memref<2x16xi32>, %out_bx1: memref<2x16xi32>,
      %out_ax0: memref<2x16xi32>, %out_ax1: memref<2x16xi32>,
      %in_bx0: memref<2x16xi32>, %in_bx1: memref<2x16xi32>,
      %in_ax0: memref<2x16xi32>, %in_ax1: memref<2x16xi32>,
      %consts_slab: memref<?xi8>) {
    cheddar_gpu.hoist_gs_fused
      out_bx(%out_bx0, %out_bx1),
      out_ax(%out_ax0, %out_ax1),
      in_bx(%in_bx0, %in_bx1),
      in_ax(%in_ax0, %in_ax1),
      %consts_slab <{
        num_aux = 1 : i64,
        num_q = 1 : i64,
        prime_offset = 0 : i64,
        pt_limbs = 2 : i64,
        pt_word_offsets = array<i64: 0, 32, -1, 64>
      }> : (memref<2x16xi32>, memref<2x16xi32>,
            memref<2x16xi32>, memref<2x16xi32>,
            memref<2x16xi32>, memref<2x16xi32>,
            memref<2x16xi32>, memref<2x16xi32>,
            memref<?xi8>) -> ()
    return
  }
}
