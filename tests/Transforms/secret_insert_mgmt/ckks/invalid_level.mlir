// RUN: heir-opt --secret-insert-mgmt-ckks="after-mul=false before-mul-include-first-mul=false bootstrap-waterline=0 level-budget=1 min-slot-count=4096" %s | FileCheck %s

module attributes {backend.lattigo, scheme.ckks, backend.config_override = {bootstrapLevelsConsumed = 2 : i32}} {
  func.func @reproducer(%arg0: !secret.secret<tensor<1x4096xf32>>) -> !secret.secret<tensor<1x4096xf32>> {
    %cst = arith.constant dense<1.000000e+00> : tensor<1x4096xf32>
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<1x4096xf32>
    %cst_1 = arith.constant dense<3.000000e+00> : tensor<1x4096xf32>
    %cst_2 = arith.constant dense<4.000000e+00> : tensor<1x4096xf32>
    %cst_3 = arith.constant dense<5.000000e+00> : tensor<1x4096xf32>
    %cst_4 = arith.constant dense<6.000000e+00> : tensor<1x4096xf32>
    %cst_5 = arith.constant dense<7.000000e+00> : tensor<1x4096xf32>
    %cst_6 = arith.constant dense<8.000000e+00> : tensor<1x4096xf32>
    %cst_7 = arith.constant dense<9.000000e+00> : tensor<1x4096xf32>
    %cst_8 = arith.constant dense<1.000000e+01> : tensor<1x4096xf32>
    %cst_9 = arith.constant dense<1.100000e+01> : tensor<1x4096xf32>
    %cst_10 = arith.constant dense<1.200000e+01> : tensor<1x4096xf32>
    %cst_11 = arith.constant dense<1.300000e+01> : tensor<1x4096xf32>
    %cst_12 = arith.constant dense<1.400000e+01> : tensor<1x4096xf32>
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %c2048 = arith.constant 2048 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index

    %4534 = secret.generic(%arg0: !secret.secret<tensor<1x4096xf32>> {heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 1, 10, 48>}}) {
    ^body(%input0: tensor<1x4096xf32>):
      %11354 = arith.mulf %input0, %cst : tensor<1x4096xf32>
      %11355 = arith.addf %11354, %cst_0 : tensor<1x4096xf32>
      %11356 = arith.mulf %11355, %cst_1 : tensor<1x4096xf32>
      %11357 = arith.mulf %11355, %cst_3 : tensor<1x4096xf32>
      %11358 = arith.addf %11357, %cst_4 : tensor<1x4096xf32>
      %11359 = arith.mulf %11355, %11355 : tensor<1x4096xf32>
      %11360 = arith.addf %11359, %11359 : tensor<1x4096xf32>
      %11361 = arith.subf %11360, %cst_5 : tensor<1x4096xf32>
      %11362 = arith.mulf %11358, %11361 : tensor<1x4096xf32>
      %11363 = arith.addf %11356, %cst_2 : tensor<1x4096xf32>
      %11364 = arith.addf %11363, %11362 : tensor<1x4096xf32>
      %11365 = arith.mulf %11364, %cst_6 : tensor<1x4096xf32>
      %11366 = arith.mulf %11364, %cst_7 : tensor<1x4096xf32>
      %11367 = tensor_ext.rotate %11366, %c1 : tensor<1x4096xf32>, index
      %11368 = arith.addf %11365, %11367 : tensor<1x4096xf32>
      %11369 = arith.mulf %11368, %cst_8 : tensor<1x4096xf32>
      %11370 = arith.mulf %11368, %cst_9 : tensor<1x4096xf32>
      %11371 = tensor_ext.rotate %11370, %c2 : tensor<1x4096xf32>, index
      %11372 = arith.addf %11369, %11371 : tensor<1x4096xf32>
      %11373 = arith.mulf %11372, %cst_10 : tensor<1x4096xf32>
      %11374 = arith.mulf %11372, %cst_11 : tensor<1x4096xf32>
      %11375 = tensor_ext.rotate %11374, %c4 : tensor<1x4096xf32>, index
      %11376 = arith.addf %11373, %11375 : tensor<1x4096xf32>
      %11377 = arith.mulf %11376, %cst_12 : tensor<1x4096xf32>
      %extra = arith.mulf %11377, %cst_12 : tensor<1x4096xf32>
      // CHECK-COUNT-5: mgmt.bootstrap
      secret.yield %extra : tensor<1x4096xf32>
    } -> !secret.secret<tensor<1x4096xf32>>
    return %4534 : !secret.secret<tensor<1x4096xf32>>
  }
}
