// RUN: heir-opt --rotation-analysis %s | FileCheck %s

// CHECK: module attributes
// CHECK-SAME: rotation_analysis.indices = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 46, 69, 92, 115, 138, 161, 184, 207, 230, 253, 276, 299, 322, 345, 368, 391, 414, 437, 460, 483, 506, 512, 518, 541, 564, 587, 610, 633, 656, 679, 702, 725, 748, 771, 794, 817, 840, 863, 886, 909, 932, 955, 978, 1001>
module {
  func.func @matvec(
      %arg0: !secret.secret<tensor<1x1024xf32>>) -> (!secret.secret<tensor<1x1024xf32>>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %c0 = arith.constant 0 : index
    %c23 = arith.constant 23 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c-23 = arith.constant -23 : index
    %0 = arith.constant dense<2.0> : tensor<512x1024xf32>
    %1 = secret.generic(%arg0: !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %2 = scf.for %arg1 = %c0 to %c23 step %c1 iter_args(%arg2 = %cst) -> (tensor<1x1024xf32>) {
        %6 = scf.for %arg3 = %c0 to %c23 step %c1 iter_args(%arg4 = %cst) -> (tensor<1x1024xf32>) {
          %10 = arith.muli %arg1, %c23 : index
          %11 = arith.addi %arg3, %10 : index
          %12 = arith.cmpi slt, %11, %c512 : index
          %13 = scf.if %12 -> (tensor<1x1024xf32>) {
            %extracted_slice = tensor.extract_slice %0[%11, 0] [1, 1024] [1, 1] : tensor<512x1024xf32> to tensor<1x1024xf32>
            %14 = arith.muli %arg1, %c-23 : index
            %15 = tensor_ext.rotate %extracted_slice, %14 : tensor<1x1024xf32>, index
            %16 = tensor_ext.rotate %input0, %arg3 : tensor<1x1024xf32>, index
            %17 = arith.mulf %15, %16 : tensor<1x1024xf32>
            %18 = arith.addf %arg4, %17 : tensor<1x1024xf32>
            scf.yield %18 : tensor<1x1024xf32>
          } else {
            scf.yield %arg4 : tensor<1x1024xf32>
          }
          scf.yield %13 : tensor<1x1024xf32>
        }
        %7 = arith.muli %arg1, %c23 : index
        %8 = tensor_ext.rotate %6, %7 : tensor<1x1024xf32>, index
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        scf.yield %9 : tensor<1x1024xf32>
      }
      %3 = tensor_ext.rotate %2, %c512 : tensor<1x1024xf32>, index
      %4 = arith.addf %2, %3 : tensor<1x1024xf32>
      %5 = arith.addf %4, %cst : tensor<1x1024xf32>
      secret.yield %5 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %1 : !secret.secret<tensor<1x1024xf32>>
  }
}
