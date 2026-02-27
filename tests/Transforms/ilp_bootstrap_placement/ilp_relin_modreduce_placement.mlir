// RUN: heir-opt --ilp-bootstrap-placement=bootstrap-waterline=3 %s | FileCheck %s

// Test relinearize and modreduce operations are inserted correctly after the 
// ILP bootstrap placement pass, i.e., each multiplication should be followed by
// a `mgmt.relinearize` and a `mgmt.modreduce`. This test samples 3 mul ops and 
// checks that these mgmt operations are inserted.

// CHECK: func.func @bootstrap_placement_test
// CHECK: secret.generic
// CHECK: arith.mulf %input3, %input4
// CHECK-NEXT: mgmt.relinearize
// CHECK-NEXT: mgmt.modreduce
// CHECK: arith.mulf %3, %6
// CHECK-NEXT: mgmt.relinearize
// CHECK-NEXT: mgmt.modreduce
// CHECK: arith.mulf %10, %14
// CHECK-NEXT: mgmt.relinearize
// CHECK-NEXT: mgmt.modreduce
// CHECK: secret.yield


func.func @bootstrap_placement_test(
    %arg0: !secret.secret<tensor<8xf32>>,
    %arg1: !secret.secret<tensor<8xf32>>,
    %arg2: !secret.secret<tensor<8xf32>>,
    %arg3: !secret.secret<tensor<8xf32>>,
    %arg4: !secret.secret<tensor<8xf32>>) -> !secret.secret<tensor<8xf32>> {
  %0 = secret.generic(
      %arg0: !secret.secret<tensor<8xf32>>,
      %arg1: !secret.secret<tensor<8xf32>>,
      %arg2: !secret.secret<tensor<8xf32>>,
      %arg3: !secret.secret<tensor<8xf32>>,
      %arg4: !secret.secret<tensor<8xf32>>) {
  ^body(%input0: tensor<8xf32>,
        %input1: tensor<8xf32>,
        %input2: tensor<8xf32>,
        %input3: tensor<8xf32>,
        %input4: tensor<8xf32>):
    %l1 = arith.mulf %input0, %input1 : tensor<8xf32>
    %l2 = arith.mulf %input1, %input2 : tensor<8xf32>
    %l3 = arith.mulf %input3, %input4 : tensor<8xf32>
    %l4 = arith.mulf %l1, %l2 : tensor<8xf32>
    %l5 = arith.mulf %input0, %l4 : tensor<8xf32>
    %l6 = arith.mulf %l4, %input3 : tensor<8xf32>
    %l7 = arith.mulf %l3, %l4 : tensor<8xf32>
    %l8 = arith.mulf %l5, %l6 : tensor<8xf32>
    %l9 = arith.mulf %l6, %l7 : tensor<8xf32>
    %l10 = arith.mulf %l8, %l9 : tensor<8xf32>
    secret.yield %l10 : tensor<8xf32>
  } -> !secret.secret<tensor<8xf32>>
  return %0 : !secret.secret<tensor<8xf32>>
}
