// RUN: heir-opt --optimize-relinearization %s | FileCheck %s

// CHECK: func.func @repro
// CHECK-COUNT-1: mgmt.relinearize
// CHECK-NOT: mgmt.relinearize
func.func @repro(%arg0: !secret.secret<tensor<1024xi16>>, %arg1: !secret.secret<tensor<1024xi16>>, %arg2: i16) -> (!secret.secret<tensor<1024xi16>>) {
  %splat = tensor.splat %arg2 : tensor<1024xi16>
  %0 = secret.generic(%arg0: !secret.secret<tensor<1024xi16>>, %arg1: !secret.secret<tensor<1024xi16>>) {
  ^body(%input0: tensor<1024xi16>, %input1: tensor<1024xi16>):
    %1 = arith.muli %input0, %input0 : tensor<1024xi16>
    %2 = arith.muli %input1, %input1 : tensor<1024xi16>
    %3 = arith.muli %input0, %splat : tensor<1024xi16>
    %4 = arith.addi %3, %1 : tensor<1024xi16>
    %5 = arith.addi %4, %2 : tensor<1024xi16>
    secret.yield %5 : tensor<1024xi16>
  } -> !secret.secret<tensor<1024xi16>>
  return %0 : !secret.secret<tensor<1024xi16>>
}

// CHECK: func.func @repro2
// CHECK-COUNT-1: mgmt.relinearize
// CHECK-NOT: mgmt.relinearize
func.func @repro2(%arg0: !secret.secret<tensor<1024xi16>>, %arg1: !secret.secret<tensor<1024xi16>>, %arg2: i16) -> (!secret.secret<tensor<1024xi16>>) {
  %splat = tensor.splat %arg2 : tensor<1024xi16>
  %0 = secret.generic(%arg0: !secret.secret<tensor<1024xi16>>, %arg1: !secret.secret<tensor<1024xi16>>) {
  ^body(%input0: tensor<1024xi16>, %input1: tensor<1024xi16>):
    %1 = arith.muli %input0, %input0 : tensor<1024xi16>
    %2 = arith.muli %input1, %input1 : tensor<1024xi16>
    %3 = arith.muli %input0, %splat : tensor<1024xi16>
    %4 = arith.addi %1, %2 : tensor<1024xi16>
    %5 = arith.addi %4, %3 : tensor<1024xi16>
    secret.yield %5 : tensor<1024xi16>
  } -> !secret.secret<tensor<1024xi16>>
  return %0 : !secret.secret<tensor<1024xi16>>
}
