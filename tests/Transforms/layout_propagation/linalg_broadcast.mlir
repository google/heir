// RUN: heir-opt --layout-propagation --split-input-file %s | FileCheck %s
// This tests linalg broadcast layout assignment with a torch-mlir layernorm sample.

// CHECK-DAG: #layout = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : (256i1 - i2 + slot + 1024*floor((-256i1 + i2)/1024)) mod 131072 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 63 and 0 <= i2 <= 767 and 0 <= ct <= 95 and -1023 + 49152i0 + 768i1 + i2 <= 1024ct <= 49152i0 + 768i1 + i2 and 0 <= slot <= 1023 }">
// CHECK-DAG: #layout1 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (256i1 + slot + 1024*floor((-i1)/4)) mod 131072 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 63 and -3 + 192i0 + 3i1 <= 4ct <= 192i0 + 3i1 and 0 <= slot <= 1023 }">
module {
  // CHECK: func @main
  func.func @main(%arg0: !secret.secret<tensor<2x64x768xf32>> {secret.secret}) -> (!secret.secret<tensor<2x64x768xf32>> {secret.secret}) {
    %cst = arith.constant dense<-1210308.13> : tensor<2x64xf32>
    %cst_0 = arith.constant dense<-6766.85888> : tensor<2x64xf32>
    %cst_1 = arith.constant dense<611920.813> : tensor<2x64xf32>
    %cst_2 = arith.constant dense<2.000000e+00> : tensor<2x64xf32>
    %cst_3 = arith.constant dense<-6766.6372> : tensor<2x64xf32>
    %cst_4 = arith.constant dense<6766.38086> : tensor<2x64xf32>
    %cst_5 = arith.constant dense<-3.382790e+03> : tensor<2x64xf32>
    %cst_6 = arith.constant dense<1.000000e+00> : tensor<2x64xf32>
    %cst_7 = arith.constant dense<2.000000e-01> : tensor<2x64xf32>
    %cst_8 = arith.constant dense<9.9999999999999998E-13> : tensor<2x64xf64>
    %cst_9 = arith.constant dense<7.680000e+02> : tensor<2x64xf32>
    %cst_10 = arith.constant dense<0.000000e+00> : tensor<2x64xf32>
    %0 = tensor.empty() {secret.public} : tensor<2x64x768xf32>
    %1 = arith.truncf %cst_8 {secret.public} : tensor<2x64xf64> to tensor<2x64xf32>
    %2 = secret.generic(%arg0: !secret.secret<tensor<2x64x768xf32>>) attrs = {secret.secret} {
    ^body(%input0: tensor<2x64x768xf32>):
      // CHECK-DAG linalg.reduce ins(%input0 : tensor<2x64x768xf32>) outs(%cst_10 : tensor<2x64xf32>) dimensions = [2]  {secret.secret} tensor_ext.layout = [[#layout1]]
      %reduced = linalg.reduce ins(%input0 : tensor<2x64x768xf32>) outs(%cst_10 : tensor<2x64xf32>) dimensions = [2]  {secret.secret}
        (%in: f32, %init: f32) {
          %30 = arith.addf %in, %init {secret.public} : f32
          linalg.yield %30 {secret.public} : f32
        }
      // CHECK-DAG arith.divf %reduced, %cst_9 {secret.secret} : tensor<2x64xf32> {tensor_ext.layout = [[#layout1]]}
      %3 = arith.divf %reduced, %cst_9 {secret.secret} : tensor<2x64xf32>
      // CHECK-DAG linalg.broadcast ins(%3 : tensor<2x64xf32>) outs(%0 : tensor<2x64x768xf32>) dimensions = [2]  {secret.secret} tensor_ext.layout = [[#layout]]
      %broadcasted = linalg.broadcast ins(%3 : tensor<2x64xf32>) outs(%0 : tensor<2x64x768xf32>) dimensions = [2]  {secret.secret}
      // CHECK-DAG arith.subf %input0, %broadcasted {secret.secret} : tensor<2x64x768xf32> {tensor_ext.layout = [[#layout]]}
      %4 = arith.subf %input0, %broadcasted {secret.secret} : tensor<2x64x768xf32>
      %5 = arith.mulf %4, %4 {secret.secret} : tensor<2x64x768xf32>
      // CHECK-DAG linalg.reduce ins(%5 : tensor<2x64x768xf32>) outs(%cst_10 : tensor<2x64xf32>) dimensions = [2]  {secret.secret} tensor_ext.layout = [[#layout1]]
      %reduced_11 = linalg.reduce ins(%5 : tensor<2x64x768xf32>) outs(%cst_10 : tensor<2x64xf32>) dimensions = [2]  {secret.secret}
        (%in: f32, %init: f32) {
          %30 = arith.addf %in, %init {secret.public} : f32
          linalg.yield %30 {secret.public} : f32
        }
      // CHECK-DAG arith.divf %reduced_11, %cst_9 {secret.secret} : tensor<2x64xf32> {tensor_ext.layout = [[#layout1]]}
      %6 = arith.divf %reduced_11, %cst_9 {secret.secret} : tensor<2x64xf32>
      %7 = arith.addf %6, %1 {secret.secret} : tensor<2x64xf32>
      %8 = arith.mulf %7, %cst_7 : tensor<2x64xf32>
      %9 = arith.addf %8, %cst_6 : tensor<2x64xf32>
      %10 = arith.mulf %9, %cst_4 : tensor<2x64xf32>
      %11 = arith.addf %10, %cst_5 : tensor<2x64xf32>
      %12 = arith.mulf %9, %9 : tensor<2x64xf32>
      %13 = arith.mulf %12, %cst_2 : tensor<2x64xf32>
      %14 = arith.subf %13, %cst_6 : tensor<2x64xf32>
      %15 = arith.mulf %14, %cst_3 : tensor<2x64xf32>
      %16 = arith.addf %11, %15 : tensor<2x64xf32>
      %17 = arith.mulf %9, %14 : tensor<2x64xf32>
      %18 = arith.mulf %17, %cst_2 : tensor<2x64xf32>
      %19 = arith.subf %18, %9 : tensor<2x64xf32>
      %20 = arith.mulf %19, %cst_1 : tensor<2x64xf32>
      %21 = arith.addf %16, %20 : tensor<2x64xf32>
      %22 = arith.mulf %9, %cst : tensor<2x64xf32>
      %23 = arith.addf %22, %cst_0 : tensor<2x64xf32>
      %24 = arith.mulf %14, %14 : tensor<2x64xf32>
      %25 = arith.mulf %24, %cst_2 : tensor<2x64xf32>
      %26 = arith.subf %25, %cst_6 : tensor<2x64xf32>
      %27 = arith.mulf %23, %26 : tensor<2x64xf32>
      %28 = arith.addf %21, %27 : tensor<2x64xf32>
      // CHECK-DAG linalg.broadcast ins(%28 : tensor<2x64xf32>) outs(%0 : tensor<2x64x768xf32>) dimensions = [2]  {secret.secret} tensor_ext.layout = [[#layout]]
      %broadcasted_12 = linalg.broadcast ins(%28 : tensor<2x64xf32>) outs(%0 : tensor<2x64x768xf32>) dimensions = [2]  {secret.secret}
      // CHECK-DAG arith.mulf %broadcasted_12, %cst_9 {secret.secret} : tensor<2x64x768xf32> {tensor_ext.layout = [[#layout]]}
      %29 = arith.mulf %4, %broadcasted_12 {secret.secret} : tensor<2x64x768xf32>
      secret.yield %29 {secret.secret} : tensor<2x64x768xf32>
    } -> !secret.secret<tensor<2x64x768xf32>>
    return {secret.secret} %2 : !secret.secret<tensor<2x64x768xf32>>
  }
}
