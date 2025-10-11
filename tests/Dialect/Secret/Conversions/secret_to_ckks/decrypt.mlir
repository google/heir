// RUN: heir-opt --secret-to-ckks %s | FileCheck %s

#layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : i0 = 0 and ct = 0 and (-i1 + slot) mod 16 = 0 and 0 <= i1 <= 9 and 0 <= slot <= 1023 }">
#original_type = #tensor_ext.original_type<originalType = tensor<1x10xf32>, layout = #layout>
module @jit_func attributes {ckks.schemeParam = #ckks.scheme_param<logN = 15, Q = [36028797019488257, 35184372744193, 35184373006337, 35184373989377, 35184376545281, 35184377331713, 35184378511361, 35184378707969], P = [36028797020209153, 36028797020602369, 36028797020864513], logDefaultScale = 45>, jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32, scheme.ckks} {
  func.func public @main(%arg0: !secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 7, scale = 45>, tensor_ext.original_type = #original_type}) -> (!secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 7, scale = 45>, tensor_ext.original_type = #original_type}) {
    return %arg0 : !secret.secret<tensor<1x1024xf32>>
  }
  // CHECK: func.func @main__decrypt__result0
  // CHECK: tensor.extract
  // CHECK: lwe.rlwe_decrypt
  // CHECK: lwe.rlwe_decode
  // CHECK: tensor.concat
  func.func @main__decrypt__result0(%arg0: !secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 7, scale = 45>}) -> tensor<1x10xf32> attributes {client.dec_func = {func_name = "main", index = 0 : i64}} {
    %c1024 = arith.constant 1024 : index
    %c16 = arith.constant 16 : index
    %c6 = arith.constant 6 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<1x10xf32>
    %0 = secret.reveal %arg0 : !secret.secret<tensor<1x1024xf32>> -> tensor<1x1024xf32>
    %1 = scf.for %arg1 = %c0 to %c1024 step %c1 iter_args(%arg2 = %cst) -> (tensor<1x10xf32>) {
      %2 = arith.addi %arg1, %c6 : index
      %3 = arith.remsi %2, %c16 : index
      %4 = arith.cmpi sge, %3, %c6 : index
      %5 = scf.if %4 -> (tensor<1x10xf32>) {
        %6 = arith.remsi %arg1, %c16 : index
        %extracted = tensor.extract %0[%c0, %arg1] : tensor<1x1024xf32>
        %inserted = tensor.insert %extracted into %arg2[%c0, %6] : tensor<1x10xf32>
        scf.yield %inserted : tensor<1x10xf32>
      } else {
        scf.yield %arg2 : tensor<1x10xf32>
      }
      scf.yield %5 : tensor<1x10xf32>
    }
    return %1 : tensor<1x10xf32>
  }
}
