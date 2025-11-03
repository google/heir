// RUN: heir-opt --annotate-mgmt %s | FileCheck %s

// CHECK: func @dimension_backprop
func.func @dimension_backprop(
    %arg0: !secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 0>},
    %arg1: !secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 0>}
  ) -> (!secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 0>}) {
  %cst = arith.constant dense<7> : tensor<1024xi16>
  // CHECK: mgmt.init %{{.*}} {mgmt.mgmt = #mgmt.mgmt<level = 0, dimension = 3, scale = 0>}
  %0 = mgmt.init %cst {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 0>} : tensor<1024xi16>
  %1 = secret.generic(
      %arg0: !secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 0>},
      %arg1: !secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 0>}
    ) {
    ^body(%input0: tensor<1024xi16>, %input1: tensor<1024xi16>):
      // CHECK: arith.muli
      // CHECK-SAME: {mgmt.mgmt = #mgmt.mgmt<level = 0, dimension = 3, scale = 0>}
      %2 = arith.muli %input0, %input1 {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 0>} : tensor<1024xi16>
      // Here, the result of the ciphertext-plaintext mul does not increase
      // the dimension of the ciphertext. All that's happening here is the
      // plaintext (mgmt.init above) is informed the level information required
      // of it to encode to a compatible plaintext.
      // CHECK: arith.muli
      // CHECK-SAME: {mgmt.mgmt = #mgmt.mgmt<level = 0, dimension = 3, scale = 0>}
      %3 = arith.muli %0, %2 {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 0>} : tensor<1024xi16>
      secret.yield %3 : tensor<1024xi16>
  } -> (!secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 0>})
  return %1 : !secret.secret<tensor<1024xi16>>
}
