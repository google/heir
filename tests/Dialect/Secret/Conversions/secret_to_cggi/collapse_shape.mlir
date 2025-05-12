// RUN: heir-opt --secret-distribute-generic --canonicalize --secret-to-cggi %s | FileCheck %s

// Ensure that we use a collapse_shape operation to reconcile tensor<2x4xlwe_ct>
// with tensor<8xlwe_ct>.

// CHECK: ![[ct_ty:.*]] = !lwe.lwe_ciphertext
// CHECK: module

module {
  // CHECK: func.func @collapse_shape
  // CHECK-SAME:    (%[[arg0:.*]]: tensor<2x4x![[ct_ty]]>) -> tensor<2x4x![[ct_ty]]>
  func.func @collapse_shape(%arg0: !secret.secret<tensor<2xi4>>) -> !secret.secret<tensor<2xi4>> {
    %c3 = arith.constant 3 : index
    %false = arith.constant false
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    // CHECK-NOT: reinterpret_cast
    // CHECK-COUNT-6: tensor.extract %[[arg0]]
    %0 = secret.cast %arg0 : !secret.secret<tensor<2xi4>> to !secret.secret<tensor<8xi1>>
    %1 = secret.generic(%0 : !secret.secret<tensor<8xi1>>) {
    ^body(%input0: tensor<8xi1>):
      %3 = tensor.extract %input0[%c0] : tensor<8xi1>
      %4 = tensor.extract %input0[%c1] : tensor<8xi1>
      %5 = tensor.extract %input0[%c2] : tensor<8xi1>
      %6 = tensor.extract %input0[%c4] : tensor<8xi1>
      %7 = tensor.extract %input0[%c5] : tensor<8xi1>
      %8 = tensor.extract %input0[%c6] : tensor<8xi1>
      %from_elements = tensor.from_elements %false, %3, %4, %5, %false, %6, %7, %8 : tensor<8xi1>
      secret.yield %from_elements : tensor<8xi1>
    } -> !secret.secret<tensor<8xi1>>
    // CHECK: tensor.expand_shape
    %2 = secret.cast %1 : !secret.secret<tensor<8xi1>> to !secret.secret<tensor<2xi4>>
    return %2 : !secret.secret<tensor<2xi4>>
  }
}
