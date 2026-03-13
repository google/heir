// RUN: heir-opt "--generate-param-ckks=encryption-technique-extended=false first-mod-bits=55 input-range=1 reduced-error=false scaling-mod-bits=45 slot-number=1024 use-public-key=true" %s | FileCheck %s

// This test is primarily checking that the first-mod-bits specification does not cause the pass to crash.
// It is a regression test until https://github.com/google/heir/issues/2754 is implemented.
// TODO(#2754): update/remove this test after the analysis is improved.

// CHECK: module
// CHECK-SAME: #ckks.scheme_param
// CHECK-SAME: logN = 13
// CHECK-SAME: Q = [{{[0-9]*}}, {{[0-9]*}}]
// CHECK-SAME: P = [{{[0-9]*}}]
// CHECK-SAME: logDefaultScale = 45

#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 512 = 0 and 0 <= i0 <= 511 and 0 <= slot <= 1023 }">
#original_type = #tensor_ext.original_type<originalType = tensor<512xf32>, layout = #layout>
module attributes {backend.lattigo, scheme.ckks} {
  func.func private @_assign_layout_17652308363902746083() attributes {client.pack_func = {func_name = "matvec"}} {
    return
  }
  func.func private @_assign_layout_1962505283396340287() -> tensor<512x1024xf32> attributes {client.pack_func = {func_name = "matvec"}} {
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
  func.func @matvec(%arg0: !secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1>, tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<784xf32>, layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 1024 = 0 and 0 <= i0 <= 783 and 0 <= slot <= 1023 }">>}) -> (!secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 0>, tensor_ext.original_type = #original_type}) {
    %c2 = arith.constant 2 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %c23 = arith.constant 23 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c-23 = arith.constant -23 : index
    %0 = call @_assign_layout_1962505283396340287() : () -> tensor<512x1024xf32>
    call @_assign_layout_17652308363902746083() : () -> ()
    %1 = mgmt.init %cst {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
    %2 = mgmt.init %cst {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
    %extracted_slice = tensor.extract_slice %0[0, 0] [1, 1024] [1, 1] : tensor<512x1024xf32> to tensor<1x1024xf32>
    %3 = mgmt.init %cst {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
    %4 = mgmt.init %extracted_slice {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf32>
    %5 = secret.generic(%arg0: !secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1>}) {
    ^body(%input0: tensor<1x1024xf32>):
      %6 = arith.mulf %4, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf32>
      %7 = mgmt.modreduce %6 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
      %8 = arith.addf %3, %7 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
      %9 = mgmt.level_reduce_min %8 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
      %10 = scf.for %arg1 = %c1 to %c23 step %c2 iter_args(%arg2 = %9) -> (tensor<1x1024xf32>) {
        %17 = mgmt.bootstrap %arg2 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf32>
        %18 = arith.cmpi slt, %arg1, %c512 : index
        %19 = scf.if %18 -> (tensor<1x1024xf32>) {
          %extracted_slice_0 = tensor.extract_slice %0[%arg1, 0] [1, 1024] [1, 1] : tensor<512x1024xf32> to tensor<1x1024xf32>
          %24 = tensor_ext.rotate %input0, %arg1 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf32>, index
          %25 = mgmt.init %extracted_slice_0 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf32>
          %26 = arith.mulf %25, %24 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf32>
          %27 = mgmt.modreduce %26 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
          %28 = mgmt.adjust_scale %17 {id = 0 : i64, mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf32>
          %29 = mgmt.modreduce %28 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
          %30 = arith.addf %29, %27 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
          scf.yield %30 : tensor<1x1024xf32>
        } else {
          scf.yield %17 : tensor<1x1024xf32>
        } {mgmt.mgmt = #mgmt.mgmt<level = 0>}
        %20 = arith.addi %arg1, %c1 : index
        %21 = arith.cmpi slt, %20, %c512 : index
        %22 = scf.if %21 -> (tensor<1x1024xf32>) {
          %extracted_slice_0 = tensor.extract_slice %0[%20, 0] [1, 1024] [1, 1] : tensor<512x1024xf32> to tensor<1x1024xf32>
          %24 = tensor_ext.rotate %input0, %20 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf32>, index
          %25 = mgmt.init %extracted_slice_0 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf32>
          %26 = arith.mulf %25, %24 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf32>
          %27 = mgmt.modreduce %26 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
          %28 = arith.addf %19, %27 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
          scf.yield %28 : tensor<1x1024xf32>
        } else {
          scf.yield %19 : tensor<1x1024xf32>
        } {mgmt.mgmt = #mgmt.mgmt<level = 0>}
        %23 = mgmt.level_reduce_min %22 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
        scf.yield %23 : tensor<1x1024xf32>
      } {mgmt.mgmt = #mgmt.mgmt<level = 0>}
      %11 = arith.addf %1, %10 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
      %12 = mgmt.level_reduce_min %11 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
      %13 = scf.for %arg1 = %c1 to %c23 step %c1 iter_args(%arg2 = %12) -> (tensor<1x1024xf32>) {
        %17 = mgmt.bootstrap %arg2 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf32>
        %18 = arith.muli %arg1, %c23 : index
        %19 = arith.cmpi slt, %18, %c512 : index
        %20 = scf.if %19 -> (tensor<1x1024xf32>) {
          %extracted_slice_0 = tensor.extract_slice %0[%18, 0] [1, 1024] [1, 1] : tensor<512x1024xf32> to tensor<1x1024xf32>
          %27 = arith.muli %arg1, %c-23 : index
          %28 = tensor_ext.rotate %extracted_slice_0, %27 : tensor<1x1024xf32>, index
          %29 = mgmt.init %28 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf32>
          %30 = arith.mulf %29, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf32>
          %31 = mgmt.modreduce %30 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
          %32 = mgmt.init %cst {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
          %33 = arith.addf %32, %31 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
          scf.yield %33 : tensor<1x1024xf32>
        } else {
          scf.yield %cst : tensor<1x1024xf32>
        } {mgmt.mgmt = #mgmt.mgmt<level = 0>}
        %21 = mgmt.level_reduce_min %20 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
        %22 = scf.for %arg3 = %c1 to %c23 step %c2 iter_args(%arg4 = %21) -> (tensor<1x1024xf32>) {
          %27 = mgmt.bootstrap %arg4 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf32>
          %28 = arith.addi %arg3, %18 : index
          %29 = arith.cmpi slt, %28, %c512 : index
          %30 = scf.if %29 -> (tensor<1x1024xf32>) {
            %extracted_slice_0 = tensor.extract_slice %0[%28, 0] [1, 1024] [1, 1] : tensor<512x1024xf32> to tensor<1x1024xf32>
            %36 = arith.muli %arg1, %c-23 : index
            %37 = tensor_ext.rotate %extracted_slice_0, %36 : tensor<1x1024xf32>, index
            %38 = tensor_ext.rotate %input0, %arg3 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf32>, index
            %39 = mgmt.init %37 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf32>
            %40 = arith.mulf %39, %38 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf32>
            %41 = mgmt.modreduce %40 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
            %42 = mgmt.adjust_scale %27 {id = 1 : i64, mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf32>
            %43 = mgmt.modreduce %42 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
            %44 = arith.addf %43, %41 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
            scf.yield %44 : tensor<1x1024xf32>
          } else {
            scf.yield %27 : tensor<1x1024xf32>
          } {mgmt.mgmt = #mgmt.mgmt<level = 0>}
          %31 = arith.addi %arg3, %c1 : index
          %32 = arith.addi %31, %18 : index
          %33 = arith.cmpi slt, %32, %c512 : index
          %34 = scf.if %33 -> (tensor<1x1024xf32>) {
            %extracted_slice_0 = tensor.extract_slice %0[%32, 0] [1, 1024] [1, 1] : tensor<512x1024xf32> to tensor<1x1024xf32>
            %36 = arith.muli %arg1, %c-23 : index
            %37 = tensor_ext.rotate %extracted_slice_0, %36 : tensor<1x1024xf32>, index
            %38 = tensor_ext.rotate %input0, %31 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf32>, index
            %39 = mgmt.init %37 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf32>
            %40 = arith.mulf %39, %38 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf32>
            %41 = mgmt.modreduce %40 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
            %42 = arith.addf %30, %41 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
            scf.yield %42 : tensor<1x1024xf32>
          } else {
            scf.yield %30 : tensor<1x1024xf32>
          } {mgmt.mgmt = #mgmt.mgmt<level = 0>}
          %35 = mgmt.level_reduce_min %34 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
          scf.yield %35 : tensor<1x1024xf32>
        } {mgmt.mgmt = #mgmt.mgmt<level = 0>}
        %23 = tensor_ext.rotate %22, %18 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>, index
        %24 = mgmt.adjust_scale %17 {id = 2 : i64, mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf32>
        %25 = arith.addf %24, %23 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
        %26 = mgmt.level_reduce_min %25 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
        scf.yield %26 : tensor<1x1024xf32>
      } {mgmt.mgmt = #mgmt.mgmt<level = 0>}
      %14 = tensor_ext.rotate %13, %c512 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>, index
      %15 = arith.addf %13, %14 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
      %16 = arith.addf %15, %2 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
      secret.yield %16 : tensor<1x1024xf32>
    } -> (!secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 0>})
    return %5 : !secret.secret<tensor<1x1024xf32>>
  }
  func.func @matvec__encrypt__arg0(%arg0: tensor<784xf32>) -> (!secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1>}) attributes {client.enc_func = {func_name = "matvec", index = 0 : i64}} {
    %c784_i32 = arith.constant 784 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg1 = %c0_i32 to %c784_i32 step %c1_i32 iter_args(%arg2 = %cst) -> (tensor<1x1024xf32>)  : i32 {
      %2 = arith.index_cast %arg1 : i32 to index
      %extracted = tensor.extract %arg0[%2] : tensor<784xf32>
      %inserted = tensor.insert %extracted into %arg2[%c0, %2] : tensor<1x1024xf32>
      scf.yield %inserted : tensor<1x1024xf32>
    }
    %1 = secret.conceal %0 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf32> -> !secret.secret<tensor<1x1024xf32>>
    return %1 : !secret.secret<tensor<1x1024xf32>>
  }
  func.func @matvec__decrypt__result0(%arg0: !secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) -> tensor<512xf32> attributes {client.dec_func = {func_name = "matvec", index = 0 : i64}} {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c512_i32 = arith.constant 512 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0 : index
    %0 = secret.reveal %arg0 : !secret.secret<tensor<1x1024xf32>> -> tensor<1x1024xf32>
    %1 = scf.for %arg1 = %c0_i32 to %c1024_i32 step %c1_i32 iter_args(%arg2 = %cst) -> (tensor<512xf32>)  : i32 {
      %2 = arith.remsi %arg1, %c512_i32 : i32
      %3 = arith.index_cast %arg1 : i32 to index
      %extracted = tensor.extract %0[%c0, %3] : tensor<1x1024xf32>
      %4 = arith.index_cast %2 : i32 to index
      %inserted = tensor.insert %extracted into %arg2[%4] : tensor<512xf32>
      scf.yield %inserted : tensor<512xf32>
    }
    return %1 : tensor<512xf32>
  }
}
