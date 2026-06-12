// RUN: heir-opt --implement-rotate-and-reduce="unroll=false" %s | FileCheck %s

// CHECK: @test_halevi_shoup_reduction
// CHECK-SAME: %[[arg0:.*]]: tensor<16xi32>, %[[arg1:.*]]: tensor<16x16xi32>
// CHECK: scf.for
// CHECK: scf.for
// CHECK: return

func.func @test_halevi_shoup_reduction(%0: tensor<16xi32>, %1: tensor<16x16xi32>) -> tensor<16xi32> {
  %2 = tensor_ext.rotate_and_reduce %0, %1 {period = 1 : index, steps = 16 : index} : (tensor<16xi32>, tensor<16x16xi32>) -> tensor<16xi32>
  return %2 : tensor<16xi32>
}
