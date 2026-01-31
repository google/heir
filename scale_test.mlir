// -----// IR Dump After GenerateParamCKKS (generate-param-ckks) //----- //
#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 8 = 0 and 0 <= i0 <= 7 and 0 <= slot <= 8191 }">
#original_type = #tensor_ext.original_type<originalType = tensor<8xf32>, layout = #layout>
module attributes {backend.openfhe, ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797017456641, 35184372744193, 35184371138561, 35184372121601], P = [1152921504607338497, 1152921504608747521], logDefaultScale = 45>, scheme.ckks} {
  func.func @loop(%arg0: !secret.secret<tensor<1x8192xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 3>, tensor_ext.original_type = #original_type}) -> (!secret.secret<tensor<1x8192xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 0>, tensor_ext.original_type = #original_type}) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.000000e+00 : f32
    %c8192_i32 = arith.constant 8192 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x8192xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<1x8192xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 3>}) {
    ^body(%input0: tensor<1x8192xf32>):
      %1 = scf.for %arg1 = %c0_i32 to %c8192_i32 step %c1_i32 iter_args(%arg2 = %cst_0) -> (tensor<1x8192xf32>)  : i32 {
        %16 = arith.index_cast %arg1 : i32 to index
        %inserted = tensor.insert %cst into %arg2[%c0, %16] : tensor<1x8192xf32>
        scf.yield %inserted : tensor<1x8192xf32>
      }
      %2 = mgmt.init %1 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<1x8192xf32>
      %3 = arith.mulf %input0, %2 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<1x8192xf32>
      %4 = mgmt.modreduce %3 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1x8192xf32>
      %5 = mgmt.init %1 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1x8192xf32>
      %6 = arith.subf %4, %5 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1x8192xf32>
      %7 = mgmt.level_reduce_min %6 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x8192xf32>
      %8 = affine.for %arg1 = 1 to 7 step 3 iter_args(%arg2 = %7) -> (tensor<1x8192xf32>) {
        %16 = mgmt.bootstrap %arg2 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<1x8192xf32>
        %17 = arith.mulf %input0, %16 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<1x8192xf32>
        %18 = mgmt.relinearize %17 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<1x8192xf32>
        %19 = mgmt.modreduce %18 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1x8192xf32>
        %20 = mgmt.init %1 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1x8192xf32>
        %21 = arith.subf %19, %20 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1x8192xf32>
        %22 = mgmt.adjust_scale %input0 {id = 0 : i64, mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<1x8192xf32>
        %23 = mgmt.modreduce %22 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1x8192xf32>
        %24 = arith.mulf %23, %21 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<1x8192xf32>
        %25 = mgmt.relinearize %24 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1x8192xf32>
        %26 = mgmt.modreduce %25 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x8192xf32>
        %27 = mgmt.init %1 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x8192xf32>
        %28 = arith.subf %26, %27 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x8192xf32>
        %29 = mgmt.level_reduce %input0 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1x8192xf32>
        %30 = mgmt.adjust_scale %29 {id = 1 : i64, mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1x8192xf32>
        %31 = mgmt.modreduce %30 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x8192xf32>
        %32 = arith.mulf %31, %28 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<1x8192xf32>
        %33 = mgmt.relinearize %32 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x8192xf32>
        %34 = mgmt.modreduce %33 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x8192xf32>
        %35 = mgmt.init %1 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x8192xf32>
        %36 = arith.subf %34, %35 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x8192xf32>
        %37 = mgmt.level_reduce_min %36 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x8192xf32>
        affine.yield %37 : tensor<1x8192xf32>
      } {__argattrs = [{mgmt.mgmt = #mgmt.mgmt<level = 0>}], __resattrs = [{mgmt.mgmt = #mgmt.mgmt<level = 0>}]}
      %9 = mgmt.bootstrap %8 {halo.invariance, mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<1x8192xf32>
      %10 = arith.mulf %input0, %9 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<1x8192xf32>
      %11 = mgmt.relinearize %10 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<1x8192xf32>
      %12 = mgmt.modreduce %11 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1x8192xf32>
      %13 = mgmt.init %1 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1x8192xf32>
      %14 = arith.subf %12, %13 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1x8192xf32>
      %15 = mgmt.level_reduce_min %14 {halo.invariance, mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x8192xf32>
      secret.yield %15 : tensor<1x8192xf32>
    } -> (!secret.secret<tensor<1x8192xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 0>})
    return %0 : !secret.secret<tensor<1x8192xf32>>
  }
}
