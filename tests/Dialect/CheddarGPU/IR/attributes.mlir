// RUN: heir-opt %s | heir-opt | FileCheck %s

// CHECK: module attributes
// CHECK-SAME: cheddar_gpu.params = #cheddar_gpu.params<word = 32, log_n = 16, ter = [1000], main = [2000, 3000], aux = [4000], level_config = {{.*}}, alpha = 1, dnum = 0>
module attributes {
  cheddar_gpu.params = #cheddar_gpu.params<word = 32, log_n = 16, ter = [1000], main = [2000, 3000], aux = [4000], level_config = [[[2 : i32, 1 : i32]]], alpha = 1, dnum = 0>
} {
  // CHECK: func.func @test_attributes()
  // CHECK-SAME: form_coeff = #cheddar_gpu.form<coeff>
  // CHECK-SAME: form_coeff_centered = #cheddar_gpu.form<coeff_centered>
  // CHECK-SAME: form_eval = #cheddar_gpu.form<eval>
  // CHECK-SAME: intt_mult_const = #cheddar_gpu.intt_epilogue<mult_const>
  // CHECK-SAME: intt_mult_const_normalize = #cheddar_gpu.intt_epilogue<mult_const_normalize>
  // CHECK-SAME: intt_none = #cheddar_gpu.intt_epilogue<none>
  func.func @test_attributes() attributes {
    form_coeff = #cheddar_gpu.form<coeff>,
    form_coeff_centered = #cheddar_gpu.form<coeff_centered>,
    form_eval = #cheddar_gpu.form<eval>,
    intt_mult_const = #cheddar_gpu.intt_epilogue<mult_const>,
    intt_mult_const_normalize = #cheddar_gpu.intt_epilogue<mult_const_normalize>,
    intt_none = #cheddar_gpu.intt_epilogue<none>
  } {
    return
  }
}
