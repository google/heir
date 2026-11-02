// RUN: heir-opt %s | heir-opt | FileCheck %s

// CHECK: func.func @test_pseudo_mod_up
// CHECK: cheddar_gpu.pseudo_mod_up %{{.*}}, %{{.*}} <level = 2, in_form = eval, out_form = eval> : memref<4x65536xi32>
func.func @test_pseudo_mod_up(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
  cheddar_gpu.pseudo_mod_up %out, %in <level = 2, in_form = eval, out_form = eval> : memref<4x65536xi32>
  return
}

// CHECK: func.func @test_mod_up
// CHECK: cheddar_gpu.mod_up(%{{.*}}, %{{.*}}), %{{.*}} <level = 1, in_form = eval, out_form = eval> : (memref<4x65536xi32>, memref<4x65536xi32>), memref<4x65536xi32>
func.func @test_mod_up(%out0: memref<4x65536xi32>, %out1: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
  cheddar_gpu.mod_up (%out0, %out1), %in <level = 1, in_form = eval, out_form = eval> : (memref<4x65536xi32>, memref<4x65536xi32>), memref<4x65536xi32>
  return
}

// CHECK: func.func @test_mod_down
// CHECK: cheddar_gpu.mod_down %{{.*}}, %{{.*}} <level = 1, in_form = eval, out_form = eval> : memref<3x65536xi32>, memref<5x65536xi32>
func.func @test_mod_down(%out: memref<3x65536xi32>, %in: memref<5x65536xi32>) {
  cheddar_gpu.mod_down %out, %in <level = 1, in_form = eval, out_form = eval> : memref<3x65536xi32>, memref<5x65536xi32>
  return
}

// CHECK: func.func @test_rescale
// CHECK: cheddar_gpu.rescale %{{.*}}, %{{.*}} <level = 2, in_form = eval, out_form = eval> : memref<2x65536xi64>, memref<3x65536xi64>
func.func @test_rescale(%out: memref<2x65536xi64>, %in: memref<3x65536xi64>) {
  cheddar_gpu.rescale %out, %in <level = 2, in_form = eval, out_form = eval> : memref<2x65536xi64>, memref<3x65536xi64>
  return
}

// CHECK: func.func @test_mod_down_rescale
// CHECK: cheddar_gpu.mod_down_rescale %{{.*}}, %{{.*}} <level = 2, in_form = eval, out_form = eval> : memref<2x65536xi64>, memref<5x65536xi64>
func.func @test_mod_down_rescale(%out: memref<2x65536xi64>, %in: memref<5x65536xi64>) {
  cheddar_gpu.mod_down_rescale %out, %in <level = 2, in_form = eval, out_form = eval> : memref<2x65536xi64>, memref<5x65536xi64>
  return
}

// CHECK: func.func @test_mod_down_epilogue_fused
// CHECK: cheddar_gpu.mod_down_epilogue %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} <pad_start = 0, pad_end = 3, in_form = eval, out_form = eval> : memref<3x65536xi32>, memref<3x65536xi32>, memref<3xi32>, memref<3x65536xi32>, memref<3xi32>
func.func @test_mod_down_epilogue_fused(%out: memref<3x65536xi32>, %ntt_bconv: memref<3x65536xi32>, %inv_p_prod: memref<3xi32>, %src2: memref<3x65536xi32>, %src2_padding: memref<3xi32>) {
  cheddar_gpu.mod_down_epilogue %out, %ntt_bconv, %inv_p_prod, %src2, %src2_padding <pad_start = 0, pad_end = 3, in_form = eval, out_form = eval> : memref<3x65536xi32>, memref<3x65536xi32>, memref<3xi32>, memref<3x65536xi32>, memref<3xi32>
  return
}

// CHECK: func.func @test_mod_down_epilogue_unfused
// CHECK: cheddar_gpu.mod_down_epilogue %{{.*}}, %{{.*}}, %{{.*}} <pad_start = 0, pad_end = 3, in_form = eval, out_form = eval> : memref<3x65536xi32>, memref<3x65536xi32>, memref<3xi32>
func.func @test_mod_down_epilogue_unfused(%out: memref<3x65536xi32>, %ntt_bconv: memref<3x65536xi32>, %inv_p_prod: memref<3xi32>) {
  cheddar_gpu.mod_down_epilogue %out, %ntt_bconv, %inv_p_prod <pad_start = 0, pad_end = 3, in_form = eval, out_form = eval> : memref<3x65536xi32>, memref<3x65536xi32>, memref<3xi32>
  return
}
