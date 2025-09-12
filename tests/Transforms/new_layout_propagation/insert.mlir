// RUN: heir-opt --new-layout-propagation=ciphertext-size=16 %s | FileCheck %s

// CHECK: #[[tensor_layout:.*]] = #tensor_ext.new_layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (4i0 - i1 + slot) mod 8 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 3 and 0 <= slot <= 15 }">

// CHECK: @tensor_insert
func.func @tensor_insert(%arg0: !secret.secret<tensor<2x4xi32>>) -> !secret.secret<tensor<2x4xi32>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: arith.constant 42
  %c42_i32 = arith.constant 42 : i32
  // The cleartext constant inserted into the tensor does not need a layout
  // CHECK-NOT: assign_layout
  %0 = secret.generic(%arg0 : !secret.secret<tensor<2x4xi32>>) {
  ^body(%input0: tensor<2x4xi32>):
    %1 = tensor.insert %c42_i32 into %input0[%c0, %c1] : tensor<2x4xi32>
    secret.yield %1 : tensor<2x4xi32>
  } -> !secret.secret<tensor<2x4xi32>>
  return %0 : !secret.secret<tensor<2x4xi32>>
}
