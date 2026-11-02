// RUN: heir-opt %s | FileCheck %s

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

  // CHECK: @test_neg
  func.func @test_neg(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.neg %{{.*}}, %{{.*}} <{{.*}}in_form = eval{{.*}}> : memref<4x65536xi32>
    cheddar_gpu.neg %out, %in <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: @test_add_const
  func.func @test_add_const(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>, %cst: memref<4xi32>) {
    // CHECK: cheddar_gpu.add_const %{{.*}}, %{{.*}}, %{{.*}} <{{.*}}in_form = eval{{.*}}> : memref<4x65536xi32>, memref<4xi32>
    cheddar_gpu.add_const %out, %in, %cst <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x65536xi32>, memref<4xi32>
    return
  }

  // CHECK: @test_sub_const
  func.func @test_sub_const(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>, %cst: memref<4xi32>) {
    // CHECK: cheddar_gpu.sub_const %{{.*}}, %{{.*}}, %{{.*}} <{{.*}}in_form = eval{{.*}}> : memref<4x65536xi32>, memref<4xi32>
    cheddar_gpu.sub_const %out, %in, %cst <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x65536xi32>, memref<4xi32>
    return
  }

  // CHECK: @test_sub_opposite_const
  func.func @test_sub_opposite_const(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>, %cst: memref<4xi32>) {
    // CHECK: cheddar_gpu.sub_opposite_const %{{.*}}, %{{.*}}, %{{.*}} <{{.*}}in_form = eval{{.*}}> : memref<4x65536xi32>, memref<4xi32>
    cheddar_gpu.sub_opposite_const %out, %in, %cst <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x65536xi32>, memref<4xi32>
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

  // CHECK: @test_mul_const
  func.func @test_mul_const(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>, %cst: memref<4xi32>) {
    // CHECK: cheddar_gpu.mul_const %{{.*}}, %{{.*}}, %{{.*}} <{{.*}}in_form = eval{{.*}}> : memref<4x65536xi32>, memref<4xi32>
    cheddar_gpu.mul_const %out, %in, %cst <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x65536xi32>, memref<4xi32>
    return
  }

  // CHECK: @test_cp_accum
  func.func @test_cp_accum(%out: memref<4x65536xi32>, %c0: memref<4x65536xi32>, %c1: memref<4x65536xi32>, %p0: memref<4x65536xi32>, %p1: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.cp_accum %{{.*}}, ct_srcs(%{{.*}}, %{{.*}}), common_srcs(%{{.*}}, %{{.*}}) <{{.*}}out_form = eval{{.*}}> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>) -> ()
    cheddar_gpu.cp_accum %out, ct_srcs(%c0, %c1), common_srcs(%p0, %p1) <{
      out_form = #cheddar_gpu.form<eval>
    }> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>) -> ()
    return
  }

  // CHECK: @test_cp_accum_add
  func.func @test_cp_accum_add(%out: memref<4x65536xi32>, %acc: memref<4x65536xi32>, %c0: memref<4x65536xi32>, %p0: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.cp_accum_add %{{.*}}, acc(%{{.*}}), ct_srcs(%{{.*}}), common_srcs(%{{.*}}) <{{.*}}out_form = eval{{.*}}> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>) -> ()
    cheddar_gpu.cp_accum_add %out, acc(%acc), ct_srcs(%c0), common_srcs(%p0) <{
      out_form = #cheddar_gpu.form<eval>
    }> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>) -> ()
    return
  }

  // CHECK: @test_tensor
  func.func @test_tensor(%b: memref<4x65536xi32>, %a: memref<4x65536xi32>, %r: memref<4x65536xi32>,
                         %s1_b: memref<4x65536xi32>, %s1_a: memref<4x65536xi32>,
                         %s2_b: memref<4x65536xi32>, %s2_a: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.tensor %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} <{{.*}}lhs_form = eval{{.*}}> : memref<4x65536xi32>
    cheddar_gpu.tensor %b, %a, %r, %s1_b, %s1_a, %s2_b, %s2_a <{
      lhs_form = #cheddar_gpu.form<eval>,
      rhs_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: @test_tensor_square
  func.func @test_tensor_square(%b: memref<4x65536xi32>, %a: memref<4x65536xi32>, %r: memref<4x65536xi32>,
                                %s_b: memref<4x65536xi32>, %s_a: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.tensor_square %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} <{{.*}}in_form = eval{{.*}}> : memref<4x65536xi32>
    cheddar_gpu.tensor_square %b, %a, %r, %s_b, %s_a <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: @test_mul_imag_unit
  func.func @test_mul_imag_unit(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>, %i_unit: memref<4xi32>) {
    // CHECK: cheddar_gpu.mul_imag_unit %{{.*}}, %{{.*}}, %{{.*}} <{{.*}}in_form = eval{{.*}}> : memref<4x65536xi32>, memref<4xi32>
    cheddar_gpu.mul_imag_unit %out, %in, %i_unit <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x65536xi32>, memref<4xi32>
    return
  }

  // CHECK: @test_permute_galois
  func.func @test_permute_galois(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.permute %{{.*}}, %{{.*}} <{{.*}}galois = 5{{.*}}> : memref<4x65536xi32>
    cheddar_gpu.permute %out, %in <{
      galois = 5 : i64,
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: @test_permute_conj
  func.func @test_permute_conj(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.permute %{{.*}}, %{{.*}} <{{.*}}conj{{.*}}> : memref<4x65536xi32>
    cheddar_gpu.permute %out, %in <{
      conj,
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: @test_permute_accum
  func.func @test_permute_accum(%out: memref<4x65536xi32>, %s0: memref<4x65536xi32>, %s1: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.permute_accum %{{.*}}, srcs(%{{.*}}, %{{.*}}) <{{.*}}galois_factors = [5, 25]{{.*}}> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>) -> ()
    cheddar_gpu.permute_accum %out, srcs(%s0, %s1) <{
      galois_factors = array<i64: 5, 25>,
      in_forms = [#cheddar_gpu.form<eval>, #cheddar_gpu.form<eval>],
      out_form = #cheddar_gpu.form<eval>
    }> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>) -> ()
    return
  }

  // CHECK: @test_ntt_basic
  func.func @test_ntt_basic(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.ntt %{{.*}}, %{{.*}} <in_form = coeff, out_form = eval, montgomery_conversion = false> : memref<4x65536xi32>
    cheddar_gpu.ntt %out, %in <{
      in_form = #cheddar_gpu.form<coeff>,
      out_form = #cheddar_gpu.form<eval>,
      montgomery_conversion = false
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: @test_ntt_mont_conv
  func.func @test_ntt_mont_conv(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.ntt %{{.*}}, %{{.*}} <in_form = coeff, out_form = eval, montgomery_conversion = true> : memref<4x65536xi32>
    cheddar_gpu.ntt %out, %in <{
      in_form = #cheddar_gpu.form<coeff>,
      out_form = #cheddar_gpu.form<eval>,
      montgomery_conversion = true
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: @test_ntt_with_limb_range_and_skip_range
  func.func @test_ntt_with_limb_range_and_skip_range(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.ntt %{{.*}}, %{{.*}} <{{.*}}skip_range = [1, 3]{{.*}}> : memref<4x65536xi32>
    cheddar_gpu.ntt %out, %in <{
      in_form = #cheddar_gpu.form<coeff>,
      out_form = #cheddar_gpu.form<eval>,
      montgomery_conversion = false,
      in_limb_range = array<i64: 0, 4>,
      out_limb_range = array<i64: 0, 4>,
      skip_range = array<i64: 1, 3>
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: @test_intt_basic
  func.func @test_intt_basic(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.intt %{{.*}}, %{{.*}} <{{.*}}epilogue = none{{.*}}> : memref<4x65536xi32>
    cheddar_gpu.intt %out, %in <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<coeff>,
      montgomery_conversion = true,
      epilogue = #cheddar_gpu.intt_epilogue<none>
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: @test_intt_mont_false
  func.func @test_intt_mont_false(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
    // CHECK: cheddar_gpu.intt %{{.*}}, %{{.*}} <{{.*}}epilogue = none{{.*}}> : memref<4x65536xi32>
    cheddar_gpu.intt %out, %in <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<coeff>,
      montgomery_conversion = false,
      epilogue = #cheddar_gpu.intt_epilogue<none>
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: @test_intt_mult_const
  func.func @test_intt_mult_const(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>, %cst: memref<4xi32>) {
    // CHECK: cheddar_gpu.intt %{{.*}}, %{{.*}}, %{{.*}} <{{.*}}epilogue = mult_const{{.*}}> : memref<4x65536xi32>, memref<4xi32>
    cheddar_gpu.intt %out, %in, %cst <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<coeff>,
      montgomery_conversion = true,
      epilogue = #cheddar_gpu.intt_epilogue<mult_const>
    }> : memref<4x65536xi32>, memref<4xi32>
    return
  }

  // CHECK: @test_intt_mult_const_normalize
  func.func @test_intt_mult_const_normalize(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>, %cst: memref<4xi32>) {
    // CHECK: cheddar_gpu.intt %{{.*}}, %{{.*}}, %{{.*}} <{{.*}}out_form = coeff_centered{{.*}}epilogue = mult_const_normalize{{.*}}> : memref<4x65536xi32>, memref<4xi32>
    cheddar_gpu.intt %out, %in, %cst <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<coeff_centered>,
      montgomery_conversion = true,
      epilogue = #cheddar_gpu.intt_epilogue<mult_const_normalize>
    }> : memref<4x65536xi32>, memref<4xi32>
    return
  }


  // CHECK: @test_i64_word
  func.func @test_i64_word(%out: memref<4x65536xi64>, %lhs: memref<4x65536xi64>, %rhs: memref<4x65536xi64>) {
    // CHECK: cheddar_gpu.add %{{.*}}, %{{.*}}, %{{.*}} <{{.*}}lhs_form = eval{{.*}}> : memref<4x65536xi64>
    cheddar_gpu.add %out, %lhs, %rhs <{
      lhs_form = #cheddar_gpu.form<eval>,
      rhs_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x65536xi64>
    return
  }
}
