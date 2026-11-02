// RUN: heir-opt %s | heir-opt | FileCheck %s

// CHECK: func.func @test_bconv_basic
// CHECK: cheddar_gpu.bconv %{{.*}}, %{{.*}}, %{{.*}} <in_form = coeff_centered, out_form = coeff> : memref<3x65536xi32>, memref<4x65536xi32>, memref<3x4xi32>
func.func @test_bconv_basic(%out: memref<3x65536xi32>, %in: memref<4x65536xi32>, %table: memref<3x4xi32>) {
  cheddar_gpu.bconv %out, %in, %table <in_form = coeff_centered, out_form = coeff> : memref<3x65536xi32>, memref<4x65536xi32>, memref<3x4xi32>
  return
}

// CHECK: func.func @test_bconv_with_skip
// CHECK: cheddar_gpu.bconv %{{.*}}, %{{.*}}, %{{.*}} <in_form = coeff_centered, out_form = coeff, skip_range = [2, 4]> : memref<6x65536xi32>, memref<4x65536xi32>, memref<4x4xi32>
func.func @test_bconv_with_skip(%out: memref<6x65536xi32>, %in: memref<4x65536xi32>, %table: memref<4x4xi32>) {
  cheddar_gpu.bconv %out, %in, %table <in_form = coeff_centered, out_form = coeff, skip_range = [2, 4]> : memref<6x65536xi32>, memref<4x65536xi32>, memref<4x4xi32>
  return
}

// CHECK: func.func @test_bconv_with_ranges
// CHECK: cheddar_gpu.bconv %{{.*}}, %{{.*}}, %{{.*}} <in_form = coeff_centered, out_form = coeff, in_limb_range = [1, 2], out_limb_range = [0, 3]> : memref<8x65536xi64>, memref<8x65536xi64>, memref<3x2xi64>
func.func @test_bconv_with_ranges(%out: memref<8x65536xi64>, %in: memref<8x65536xi64>, %table: memref<3x2xi64>) {
  cheddar_gpu.bconv %out, %in, %table <in_form = coeff_centered, out_form = coeff, in_limb_range = [1, 2], out_limb_range = [0, 3]> : memref<8x65536xi64>, memref<8x65536xi64>, memref<3x2xi64>
  return
}
