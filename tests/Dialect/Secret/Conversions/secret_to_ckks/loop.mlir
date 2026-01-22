// RUN: heir-opt --secret-distribute-generic --canonicalize --secret-to-ckks %s | FileCheck %s

// This test was extracted from just before distribute-generic in the e2e loop test.

#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 8 = 0 and 0 <= i0 <= 7 and 0 <= slot <= 7 }">
#original_type = #tensor_ext.original_type<originalType = tensor<8xf32>, layout = #layout>
module attributes {backend.openfhe, ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [1152921504606748673, 36028797019488257, 36028797017456641, 36028797019389953], P = [1152921504607338497, 1152921504608747521], logDefaultScale = 55>, scheme.ckks} {
  func.func private @_assign_layout_13348573087261549848(%arg0: tensor<8xf32>) -> tensor<1x8xf32> attributes {client.pack_func = {func_name = "loop"}} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<1x8xf32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = scf.for %arg1 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg2 = %cst) -> (tensor<1x8xf32>)  : i32 {
      %1 = arith.index_cast %arg1 : i32 to index
      %extracted = tensor.extract %arg0[%1] : tensor<8xf32>
      %inserted = tensor.insert %extracted into %arg2[%c0, %1] : tensor<1x8xf32>
      scf.yield %inserted : tensor<1x8xf32>
    }
    return %0 : tensor<1x8xf32>
  }

  // CHECK: @loop
  func.func @loop(%arg0: !secret.secret<tensor<1x8xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 55>, tensor_ext.original_type = #original_type}) -> (!secret.secret<tensor<1x8xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 55>, tensor_ext.original_type = #original_type}) {
    %cst = arith.constant dense<1.000000e+00> : tensor<1x8xf32>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<8xf32>
    %0 = call @_assign_layout_13348573087261549848(%cst_0) : (tensor<8xf32>) -> tensor<1x8xf32>
    %1 = mgmt.init %0 {mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 55>} : tensor<1x8xf32>
    %2 = mgmt.init %0 {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 55>} : tensor<1x8xf32>
    %3 = mgmt.init %0 {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 55>} : tensor<1x8xf32>
    %4 = secret.generic(%arg0: !secret.secret<tensor<1x8xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 55>}) {
    ^body(%input0: tensor<1x8xf32>):
      %5 = arith.mulf %input0, %1 {mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 110>} : tensor<1x8xf32>
      %6 = mgmt.modreduce %5 {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 55>} : tensor<1x8xf32>
      %7 = arith.subf %6, %2 {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 55>} : tensor<1x8xf32>
      %8 = mgmt.level_reduce %7 {levelToDrop = 2 : i64, mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 55>} : tensor<1x8xf32>

      // CHECK: affine.for
      %9 = affine.for %arg1 = 1 to 7 step 3 iter_args(%arg2 = %8) -> (tensor<1x8xf32>) {
        %16 = mgmt.bootstrap %arg2 {mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 55>} : tensor<1x8xf32>
        %17 = arith.mulf %input0, %16 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3, scale = 110>} : tensor<1x8xf32>
        %18 = mgmt.relinearize %17 {mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 110>} : tensor<1x8xf32>
        %19 = mgmt.modreduce %18 {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 55>} : tensor<1x8xf32>
        %20 = mgmt.init %0 {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 55>} : tensor<1x8xf32>
        %21 = arith.subf %19, %20 {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 55>} : tensor<1x8xf32>
        %22 = mgmt.init %cst {mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 55>} : tensor<1x8xf32>
        %23 = arith.mulf %input0, %22 {mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 110>} : tensor<1x8xf32>
        %24 = mgmt.modreduce %23 {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 55>} : tensor<1x8xf32>
        %25 = arith.mulf %24, %21 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3, scale = 110>} : tensor<1x8xf32>
        %26 = mgmt.relinearize %25 {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 110>} : tensor<1x8xf32>
        %27 = mgmt.modreduce %26 {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 55>} : tensor<1x8xf32>
        %28 = mgmt.init %0 {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 55>} : tensor<1x8xf32>
        %29 = arith.subf %27, %28 {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 55>} : tensor<1x8xf32>
        %30 = mgmt.level_reduce %input0 {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 55>} : tensor<1x8xf32>
        %31 = mgmt.init %cst {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 55>} : tensor<1x8xf32>
        %32 = arith.mulf %30, %31 {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 110>} : tensor<1x8xf32>
        %33 = mgmt.modreduce %32 {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 55>} : tensor<1x8xf32>
        %34 = arith.mulf %33, %29 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3, scale = 110>} : tensor<1x8xf32>
        %35 = mgmt.relinearize %34 {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 110>} : tensor<1x8xf32>
        %36 = mgmt.modreduce %35 {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 55>} : tensor<1x8xf32>
        %37 = mgmt.init %0 {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 55>} : tensor<1x8xf32>
        %38 = arith.subf %36, %37 {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 55>} : tensor<1x8xf32>
        affine.yield %38 : tensor<1x8xf32>
      } {__argattrs = [{mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 55>}], __resattrs = [{mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 55>}]}
      %10 = mgmt.bootstrap %9 {halo.invariance, mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 55>} : tensor<1x8xf32>
      %11 = arith.mulf %input0, %10 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3, scale = 110>} : tensor<1x8xf32>
      %12 = mgmt.relinearize %11 {mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 110>} : tensor<1x8xf32>
      %13 = mgmt.modreduce %12 {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 55>} : tensor<1x8xf32>
      %14 = arith.subf %13, %3 {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 55>} : tensor<1x8xf32>
      %15 = mgmt.level_reduce %14 {levelToDrop = 2 : i64, mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 55>} : tensor<1x8xf32>
      secret.yield %15 : tensor<1x8xf32>
    } -> (!secret.secret<tensor<1x8xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 55>})
    return %4 : !secret.secret<tensor<1x8xf32>>
  }
}
