// RUN: heir-opt --canonicalize --secret-distribute-generic --canonicalize %s | FileCheck %s

// CHECK: test_linalg_hoist_plaintext
#layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : i0 = 0 and ct = 0 and (-i1 + slot) mod 16 = 0 and 0 <= i1 <= 9 and 0 <= slot <= 1023 }">
#original_type = #tensor_ext.original_type<originalType = tensor<1x10xf32>, layout = #layout>
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 15, Q = [36028797019488257, 35184372744193, 35184373006337, 35184373989377, 35184376545281, 35184377331713, 35184378511361, 35184378707969, 35184379035649, 35184380870657, 35184381591553], P = [36028797020209153, 36028797020602369, 36028797020864513, 36028797023420417], logDefaultScale = 45>, scheme.ckks} {
  func.func @test_linalg_hoist_plaintext(%arg0: !secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 10, scale = 90>}, %arg1: !secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 10, scale = 45>, tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<1x1x32x32xf32>, layout = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and i1 = 0 and ct = 0 and (-32i2 - i3 + slot) mod 1024 = 0 and 0 <= i2 <= 31 and 0 <= i3 <= 31 and 0 <= slot <= 1023 }">>}) -> (!secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 10, scale = 90>, tensor_ext.original_type = #original_type}) {
    %cst = arith.constant dense_resource<torch_tensor_2_torch.float32> : tensor<2xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c28 = arith.constant 28 : index
    %c784 = arith.constant 784 : index
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %0 = tensor.empty() : tensor<1x2x28x28xf32>
    // CHECK-NOT: secret.generic
    // CHECK: linalg.broadcast
    // CHECK: scf.for
    // CHECK: mgmt.init
    // CHECK: secret.generic
    %1 = secret.generic(%arg0: !secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 10, scale = 45>}) {
    ^body(%input0: tensor<1x1024xf32>):
      %broadcasted = linalg.broadcast ins(%cst : tensor<2xf32>) outs(%0 : tensor<1x2x28x28xf32>) dimensions = [0, 2, 3]
      %2 = scf.for %arg2 = %c0 to %c784 step %c1 iter_args(%arg3 = %cst_0) -> (tensor<1x1024xf32>) {
        %5 = arith.divsi %arg2, %c28 : index
        %6 = arith.remsi %arg2, %c28 : index
        %extracted = tensor.extract %broadcasted[%c0, %c0, %5, %6] : tensor<1x2x28x28xf32>
        %inserted = tensor.insert %extracted into %arg3[%c0, %arg2] : tensor<1x1024xf32>
        scf.yield %inserted : tensor<1x1024xf32>
      }
      %3 = mgmt.init %2 {mgmt.mgmt = #mgmt.mgmt<level = 10, scale = 90>} : tensor<1x1024xf32>
      %4 = arith.addf %input0, %3 {mgmt.mgmt = #mgmt.mgmt<level = 10, scale = 90>} : tensor<1x1024xf32>
      secret.yield %4 : tensor<1x1024xf32>
    } -> (!secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 10, scale = 90>})
    return %1 : !secret.secret<tensor<1x1024xf32>>
  }
}
