// RUN: heir-opt --optimize-relinearization=allow-mixed-degree-operands=true %s | FileCheck %s

// CHECK-NOT: dimension = 4
func.func @two_mul(%arg0: !secret.secret<tensor<8xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 2>}, %arg1: !secret.secret<tensor<8xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 2>}) -> (!secret.secret<tensor<8xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 2>}) {
  %0 = secret.generic(%arg0: !secret.secret<tensor<8xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 2>}, %arg1: !secret.secret<tensor<8xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 2>}) {
  ^body(%input0: tensor<8xi16>, %input1: tensor<8xi16>):
    %1 = arith.muli %input0, %input1 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : tensor<8xi16>
    %2 = arith.muli %1, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 4>} : tensor<8xi16>
    %3 = mgmt.relinearize %2 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16>
    secret.yield %3 : tensor<8xi16>
  } -> (!secret.secret<tensor<8xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 2>})
  return %0 : !secret.secret<tensor<8xi16>>
}
