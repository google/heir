// RUN: heir-opt --annotate-mgmt %s
// Ensure no infinite loop

module attributes {backend.lattigo, scheme.ckks} {
  func.func @dot_product(%arg0: !secret.secret<tensor<8xf32>> {tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<8xf32>, layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : (i0 - slot) mod 8 = 0 and 7 >= i0 >= 0 and 7 >= slot >= 0 and ct = 0 }">>}, %arg1: !secret.secret<tensor<8xf32>> {tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<8xf32>, layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : (i0 - slot) mod 8 = 0 and 7 >= i0 >= 0 and 7 >= slot >= 0 and ct = 0 }">>}) -> (!secret.secret<tensor<8xf32>> {tensor_ext.original_type = #tensor_ext.original_type<originalType = f32, layout = #tensor_ext.layout<"{ [] -> [ct, slot] : ct = 0 and 0 <= slot <= 7 }">>}) {
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %inserted = tensor.insert %cst_0 into %cst_1[%c7] : tensor<8xf32>
    %inserted_2 = tensor.insert %cst into %cst_1[%c0] : tensor<8xf32>
    %0 = secret.conceal %inserted_2 : tensor<8xf32> -> !secret.secret<tensor<8xf32>>
    %1 = secret.generic(%arg0: !secret.secret<tensor<8xf32>>, %arg1: !secret.secret<tensor<8xf32>>, %0: !secret.secret<tensor<8xf32>>) {
    ^body(%input0: tensor<8xf32>, %input1: tensor<8xf32>, %input2: tensor<8xf32>):
      %2 = arith.mulf %input0, %input1 : tensor<8xf32>
      %3 = mgmt.relinearize %2 : tensor<8xf32>
      %inserted_3 = tensor.insert %cst into %input2[%c1] : tensor<8xf32>
      %inserted_4 = tensor.insert %cst into %inserted_3[%c2] : tensor<8xf32>
      %inserted_5 = tensor.insert %cst into %inserted_4[%c3] : tensor<8xf32>
      %inserted_6 = tensor.insert %cst into %inserted_5[%c4] : tensor<8xf32>
      %inserted_7 = tensor.insert %cst into %inserted_6[%c5] : tensor<8xf32>
      %inserted_8 = tensor.insert %cst into %inserted_7[%c6] : tensor<8xf32>
      %inserted_9 = tensor.insert %cst into %inserted_8[%c7] : tensor<8xf32>
      %4 = mgmt.init %inserted_9 : tensor<8xf32>
      %5 = arith.addf %3, %4 : tensor<8xf32>
      %6 = tensor_ext.rotate %5, %c6 : tensor<8xf32>, index
      %7 = tensor_ext.rotate %3, %c7 : tensor<8xf32>, index
      %8 = arith.addf %6, %7 : tensor<8xf32>
      %9 = arith.addf %8, %3 : tensor<8xf32>
      %10 = tensor_ext.rotate %9, %c6 : tensor<8xf32>, index
      %11 = arith.addf %10, %7 : tensor<8xf32>
      %12 = arith.addf %11, %3 : tensor<8xf32>
      %13 = tensor_ext.rotate %12, %c6 : tensor<8xf32>, index
      %14 = arith.addf %13, %7 : tensor<8xf32>
      %15 = arith.addf %14, %3 : tensor<8xf32>
      %16 = tensor_ext.rotate %15, %c7 : tensor<8xf32>, index
      %17 = arith.addf %16, %3 : tensor<8xf32>
      %18 = mgmt.init %inserted : tensor<8xf32>
      %19 = mgmt.modreduce %17 : tensor<8xf32>
      %20 = arith.mulf %18, %19 : tensor<8xf32>
      %21 = tensor_ext.rotate %20, %c7 : tensor<8xf32>, index
      %22 = mgmt.modreduce %21 : tensor<8xf32>
      secret.yield %22 : tensor<8xf32>
    } -> !secret.secret<tensor<8xf32>>
    return %1 : !secret.secret<tensor<8xf32>>
  }
}
