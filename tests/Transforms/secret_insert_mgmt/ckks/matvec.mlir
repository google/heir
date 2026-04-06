// RUN: heir-opt --secret-insert-mgmt-ckks="slot-number=1024 level-budget=2 after-mul=true" %s | FileCheck %s

// CHECK: @matvec
// CHECK-SAME: mgmt.mgmt = #mgmt.mgmt<level = 1>
#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 512 = 0 and 0 <= i0 <= 511 and 0 <= slot <= 1023 }">
#original_type = #tensor_ext.original_type<originalType = tensor<512xf32>, layout = #layout>
module attributes {backend.lattigo, scheme.ckks} {
  func.func private @_assign_layout_8634348465628479189() -> tensor<512x1024xf32> attributes {client.pack_func = {func_name = "matvec"}} {
    %cst = arith.constant 1.000000e+00 : f32
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<512x1024xf32>
    %c1024_i32 = arith.constant 1024 : i32
    %c240_i32 = arith.constant 240 : i32
    %0 = scf.for %arg0 = %c0_i32 to %c512_i32 step %c1_i32 iter_args(%arg1 = %cst_0) -> (tensor<512x1024xf32>)  : i32 {
      %1 = scf.for %arg2 = %c0_i32 to %c1024_i32 step %c1_i32 iter_args(%arg3 = %arg1) -> (tensor<512x1024xf32>)  : i32 {
        %2 = arith.addi %arg0, %arg2 : i32
        %3 = arith.addi %2, %c240_i32 : i32
        %4 = arith.remsi %3, %c1024_i32 : i32
        %5 = arith.cmpi sge, %4, %c240_i32 : i32
        %6 = scf.if %5 -> (tensor<512x1024xf32>) {
          %7 = arith.index_cast %arg0 : i32 to index
          %8 = arith.index_cast %arg2 : i32 to index
          %inserted = tensor.insert %cst into %arg3[%7, %8] : tensor<512x1024xf32>
          scf.yield %inserted : tensor<512x1024xf32>
        } else {
          scf.yield %arg3 : tensor<512x1024xf32>
        }
        scf.yield %6 : tensor<512x1024xf32>
      }
      scf.yield %1 : tensor<512x1024xf32>
    }
    return %0 : tensor<512x1024xf32>
  }
  func.func @matvec(%arg0: !secret.secret<tensor<1x1024xf32>> {tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<784xf32>, layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 1024 = 0 and 0 <= i0 <= 783 and 0 <= slot <= 1023 }">>}) -> (!secret.secret<tensor<1x1024xf32>> {tensor_ext.original_type = #original_type}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %c0 = arith.constant 0 : index
    %c23 = arith.constant 23 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c-23 = arith.constant -23 : index
    %0 = call @_assign_layout_8634348465628479189() : () -> tensor<512x1024xf32>
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
