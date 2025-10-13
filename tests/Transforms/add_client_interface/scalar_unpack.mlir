// RUN: heir-opt --add-client-interface %s | FileCheck %s

#layout = #tensor_ext.layout<"{ [] -> [ct, slot] : ct = 0 and 0 <= slot <= 1023 }">
#layout1 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : i0 = 0 and i1 = 0 and ct = 0 and 0 <= slot <= 1023 }">
#original_type = #tensor_ext.original_type<originalType = i16, layout = #layout>
module attributes {backend.lattigo} {
  // CHECK: hamming__decrypt__result0
  // CHECK-NEXT: secret.reveal
  // CHECK-NEXT: arith.constant 0
  // CHECK-NEXT: arith.constant 0
  // CHECK-NEXT: tensor.extract
  // CHECK-NEXT: return
  func.func @hamming(%arg0: !secret.secret<tensor<1x1024xi16>> {tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<1024xi16>, layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 1024 = 0 and 0 <= i0 <= 1023 and 0 <= slot <= 1023 }">>}, %arg1: !secret.secret<tensor<1x1024xi16>> {tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<1024xi16>, layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 1024 = 0 and 0 <= i0 <= 1023 and 0 <= slot <= 1023 }">>}) -> (!secret.secret<tensor<1x1024xi16>> {tensor_ext.original_type = #original_type}) {
    %c1_i16 = arith.constant 1 : i16
    %cst = arith.constant dense<0> : tensor<1x1024xi16>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %inserted = tensor.insert %c1_i16 into %cst[%c0, %c0] : tensor<1x1024xi16>
    %collapsed = tensor.collapse_shape %inserted [[0, 1]] : tensor<1x1024xi16> into tensor<1024xi16>
    %0 = secret.generic(%arg0: !secret.secret<tensor<1x1024xi16>>, %arg1: !secret.secret<tensor<1x1024xi16>>) {
    ^body(%input0: tensor<1x1024xi16>, %input1: tensor<1x1024xi16>):
      %collapsed_0 = tensor.collapse_shape %input0 [[0, 1]] : tensor<1x1024xi16> into tensor<1024xi16>
      %collapsed_1 = tensor.collapse_shape %input1 [[0, 1]] : tensor<1x1024xi16> into tensor<1024xi16>
      %1 = arith.subi %collapsed_0, %collapsed_1 : tensor<1024xi16>
      %2 = arith.muli %1, %1 : tensor<1024xi16>
      %3 = tensor_ext.rotate %2, %c512 : tensor<1024xi16>, index
      %4 = arith.addi %2, %3 : tensor<1024xi16>
      %5 = tensor_ext.rotate %4, %c256 : tensor<1024xi16>, index
      %6 = arith.addi %4, %5 : tensor<1024xi16>
      %7 = tensor_ext.rotate %6, %c128 : tensor<1024xi16>, index
      %8 = arith.addi %6, %7 : tensor<1024xi16>
      %9 = tensor_ext.rotate %8, %c64 : tensor<1024xi16>, index
      %10 = arith.addi %8, %9 : tensor<1024xi16>
      %11 = tensor_ext.rotate %10, %c32 : tensor<1024xi16>, index
      %12 = arith.addi %10, %11 : tensor<1024xi16>
      %13 = tensor_ext.rotate %12, %c16 : tensor<1024xi16>, index
      %14 = arith.addi %12, %13 : tensor<1024xi16>
      %15 = tensor_ext.rotate %14, %c8 : tensor<1024xi16>, index
      %16 = arith.addi %14, %15 : tensor<1024xi16>
      %17 = tensor_ext.rotate %16, %c4 : tensor<1024xi16>, index
      %18 = arith.addi %16, %17 : tensor<1024xi16>
      %19 = tensor_ext.rotate %18, %c2 : tensor<1024xi16>, index
      %20 = arith.addi %18, %19 : tensor<1024xi16>
      %21 = tensor_ext.rotate %20, %c1 : tensor<1024xi16>, index
      %22 = arith.addi %20, %21 : tensor<1024xi16>
      %23 = arith.muli %collapsed, %22 : tensor<1024xi16>
      %expanded = tensor.expand_shape %23 [[0, 1]] output_shape [1, 1024] : tensor<1024xi16> into tensor<1x1024xi16>
      %24 = tensor_ext.remap %expanded {permutation = #layout1} : tensor<1x1024xi16>
      secret.yield %24 : tensor<1x1024xi16>
    } -> !secret.secret<tensor<1x1024xi16>>
    return %0 : !secret.secret<tensor<1x1024xi16>>
  }
}
