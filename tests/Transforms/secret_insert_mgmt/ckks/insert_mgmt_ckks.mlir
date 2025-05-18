// RUN: heir-opt --full-loop-unroll --secret-insert-mgmt-ckks=slot-number=8 %s

// TODO(#1181): make this test work without the loop unroll step

#alignment = #tensor_ext.alignment<in = [], out = [1], insertedDims = [0]>
#layout = #tensor_ext.layout<map = (d0) -> (d0 mod 8), alignment = #alignment>
#original_type = #tensor_ext.original_type<originalType = f32, layout = #layout>
module attributes {backend.lattigo, scheme.ckks} {
  func.func @dot_product(%arg0: !secret.secret<tensor<8xf32>> {tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<8xf32>, layout = <map = (d0) -> (d0 mod 8)>>}, %arg1: !secret.secret<tensor<8xf32>> {tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<8xf32>, layout = <map = (d0) -> (d0 mod 8)>>}) -> (!secret.secret<tensor<8xf32>> {tensor_ext.original_type = #original_type}) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %inserted = tensor.insert %cst_0 into %cst_1[%c7] : tensor<8xf32>
    %0 = secret.generic(%arg0 : !secret.secret<tensor<8xf32>>, %arg1 : !secret.secret<tensor<8xf32>>) {
    ^body(%input0: tensor<8xf32>, %input1: tensor<8xf32>):
      %1 = arith.mulf %input0, %input1 : tensor<8xf32>
      %2 = affine.for %arg2 = 0 to 8 iter_args(%arg3 = %cst_1) -> (tensor<8xf32>) {
        %18 = arith.remsi %arg2, %c8 : index
        %19 = arith.cmpi slt, %18, %c0 : index
        %20 = arith.addi %18, %c8 : index
        %21 = arith.select %19, %20, %18 : index
        %inserted_2 = tensor.insert %cst into %arg3[%21] : tensor<8xf32>
        affine.yield %inserted_2 : tensor<8xf32>
      }
      %3 = arith.addf %1, %2 : tensor<8xf32>
      %4 = tensor_ext.rotate %3, %c6 : tensor<8xf32>, index
      %5 = tensor_ext.rotate %1, %c7 : tensor<8xf32>, index
      %6 = arith.addf %4, %5 : tensor<8xf32>
      %7 = arith.addf %6, %1 : tensor<8xf32>
      %8 = tensor_ext.rotate %7, %c6 : tensor<8xf32>, index
      %9 = arith.addf %8, %5 : tensor<8xf32>
      %10 = arith.addf %9, %1 : tensor<8xf32>
      %11 = tensor_ext.rotate %10, %c6 : tensor<8xf32>, index
      %12 = arith.addf %11, %5 : tensor<8xf32>
      %13 = arith.addf %12, %1 : tensor<8xf32>
      %14 = tensor_ext.rotate %13, %c7 : tensor<8xf32>, index
      %15 = arith.addf %14, %1 : tensor<8xf32>
      %16 = arith.mulf %inserted, %15 : tensor<8xf32>
      %17 = tensor_ext.rotate %16, %c7 : tensor<8xf32>, index
      secret.yield %17 : tensor<8xf32>
    } -> !secret.secret<tensor<8xf32>>
    return %0 : !secret.secret<tensor<8xf32>>
  }
}
