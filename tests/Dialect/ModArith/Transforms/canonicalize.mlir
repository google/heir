// RUN: heir-opt --canonicalize %s | FileCheck %s

!Z17_i64 = !mod_arith.int<17 : i64>
func.func @dot_product(%arg0: tensor<1024x!Z17_i64>, %arg1: tensor<1024x!Z17_i64>) -> tensor<1024x!Z17_i64> {
  %c7 = arith.constant 7 : index
  %0 = mod_arith.constant 1 : !Z17_i64
  %1 = mod_arith.constant dense<0> : tensor<1024x!Z17_i64>
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %inserted = tensor.insert %0 into %1[%c7] : tensor<1024x!Z17_i64>
  %2 = mod_arith.mul %arg0, %arg1 : tensor<1024x!Z17_i64>
  %3 = tensor_ext.rotate %2, %c4 : tensor<1024x!Z17_i64>, index
  %4 = mod_arith.add %2, %3 : tensor<1024x!Z17_i64>
  %5 = tensor_ext.rotate %4, %c2 : tensor<1024x!Z17_i64>, index
  %6 = mod_arith.add %4, %5 : tensor<1024x!Z17_i64>
  %7 = tensor_ext.rotate %6, %c1 : tensor<1024x!Z17_i64>, index
  %8 = mod_arith.add %6, %7 : tensor<1024x!Z17_i64>
  %9 = mod_arith.mul %inserted, %8 : tensor<1024x!Z17_i64>
  %10 = tensor_ext.rotate %9, %c7 : tensor<1024x!Z17_i64>, index
  return %10 : tensor<1024x!Z17_i64>
}
