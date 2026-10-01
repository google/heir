// RUN: heir-opt --secret-insert-mgmt-ckks="min-slot-count=8" %s | FileCheck %s

// scale-snu/cheddar has one canonical scale per level. A cross-level addition
// must therefore lower the higher operand with level_reduce, without an
// adjust_scale that the target cannot emit.

// CHECK: func @cross_level
// CHECK-NOT: mgmt.adjust_scale
// CHECK: mgmt.level_reduce
module attributes {backend.cheddar, scheme.ckks} {
  func.func @cross_level(
      %arg0: !secret.secret<tensor<8xf32>>,
      %arg1: !secret.secret<tensor<8xf32>>) -> !secret.secret<tensor<8xf32>> {
    %0 = secret.generic(%arg0 : !secret.secret<tensor<8xf32>>, %arg1 : !secret.secret<tensor<8xf32>>) {
    ^body(%input0: tensor<8xf32>, %input1: tensor<8xf32>):
      %m1 = arith.mulf %input0, %input1 : tensor<8xf32>
      %m2 = arith.mulf %m1, %m1 : tensor<8xf32>
      %result = arith.addf %m2, %m1 : tensor<8xf32>
      secret.yield %result : tensor<8xf32>
    } -> !secret.secret<tensor<8xf32>>
    return %0 : !secret.secret<tensor<8xf32>>
  }
}
