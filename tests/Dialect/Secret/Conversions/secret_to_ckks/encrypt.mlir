// RUN: heir-opt --secret-to-ckks %s | FileCheck %s

#new_layout = #tensor_ext.new_layout<"{ [i0, i1] -> [ct, slot] : i0 = 0 and ct = 0 and (-i1 + slot) mod 1024 = 0 and 0 <= i1 <= 783 and 0 <= slot <= 1023 }">
#original_type = #tensor_ext.original_type<originalType = tensor<1x784xf32>, layout = #new_layout>
// CHECK: ![[ct_ty:.*]] = !lwe.lwe_ciphertext
module @jit_func attributes {ckks.schemeParam = #ckks.scheme_param<logN = 15, Q = [36028797019488257, 35184372744193, 35184373006337, 35184373989377, 35184376545281, 35184377331713, 35184378511361, 35184378707969], P = [36028797020209153, 36028797020602369, 36028797020864513], logDefaultScale = 45>, jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32, scheme.ckks} {
  func.func public @main(%arg0: tensor<512x784xf32> {mhlo.sharding = "{replicated}"}, %arg1: tensor<512xf32> {mhlo.sharding = "{replicated}"}, %arg2: tensor<10x512xf32> {mhlo.sharding = "{replicated}"}, %arg3: tensor<10xf32> {mhlo.sharding = "{replicated}"}, %arg4: !secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 7, scale = 45>, tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<1x784xf32>, layout = #tensor_ext.new_layout<"{ [i0, i1] -> [ct, slot] : i0 = 0 and ct = 0 and (-i1 + slot) mod 1024 = 0 and 0 <= i1 <= 783 and 0 <= slot <= 1023 }">>}) -> (!secret.secret<tensor<1x1024xf32>> {jax.result_info = "result[0]", mgmt.mgmt = #mgmt.mgmt<level = 7, scale = 45>, tensor_ext.original_type = #original_type}) {
    return %arg4: !secret.secret<tensor<1x1024xf32>>
  }
  // CHECK: func.func @main__encrypt__arg4
  // CHECK: return %[[from_elements:.*]] : tensor<1x![[ct_ty]]>
  func.func @main__encrypt__arg4(%arg0: tensor<1x784xf32>) -> (!secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 7, scale = 45>}) attributes {client.enc_func = {func_name = "main", index = 4 : i64}} {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c784 = arith.constant 784 : index
    %0 = scf.for %arg1 = %c0 to %c784 step %c1 iter_args(%arg2 = %cst) -> (tensor<1x1024xf32>) {
      %extracted = tensor.extract %arg0[%c0, %arg1] : tensor<1x784xf32>
      %inserted = tensor.insert %extracted into %arg2[%c0, %arg1] : tensor<1x1024xf32>
      scf.yield %inserted : tensor<1x1024xf32>
    }
    %1 = secret.conceal %0 {mgmt.mgmt = #mgmt.mgmt<level = 7, scale = 45>} : tensor<1x1024xf32> -> !secret.secret<tensor<1x1024xf32>>
    return %1 : !secret.secret<tensor<1x1024xf32>>
  }
}
