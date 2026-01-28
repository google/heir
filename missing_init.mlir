#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 8 = 0 and 0 <= i0 <= 7 and 0 <= slot <= 8191 }">
#original_type = #tensor_ext.original_type<originalType = tensor<8xf32>, layout = #layout>
module attributes {backend.openfhe, scheme.ckks} {
  func.func @loop(%arg0: !secret.secret<tensor<1x8192xf32>> {tensor_ext.original_type = #original_type}) -> (!secret.secret<tensor<1x8192xf32>> {tensor_ext.original_type = #original_type}) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.000000e+00 : f32
    %c8192_i32 = arith.constant 8192 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x8192xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<1x8192xf32>>) {
    ^body(%input0: tensor<1x8192xf32>):
      %1 = scf.for %arg1 = %c0_i32 to %c8192_i32 step %c1_i32 iter_args(%arg2 = %cst_0) -> (tensor<1x8192xf32>)  : i32 {
        %3 = arith.index_cast %arg1 : i32 to index
        %inserted = tensor.insert %cst into %arg2[%c0, %3] : tensor<1x8192xf32>
        scf.yield %inserted : tensor<1x8192xf32>
      }
      %2 = affine.for %arg1 = 0 to 8 iter_args(%arg2 = %1) -> (tensor<1x8192xf32>) {
        %3 = arith.mulf %input0, %arg2 : tensor<1x8192xf32>
        %4 = arith.subf %3, %1 : tensor<1x8192xf32>
        affine.yield %4 : tensor<1x8192xf32>
      }
      secret.yield %2 : tensor<1x8192xf32>
    } -> !secret.secret<tensor<1x8192xf32>>
    return %0 : !secret.secret<tensor<1x8192xf32>>
  }
}
