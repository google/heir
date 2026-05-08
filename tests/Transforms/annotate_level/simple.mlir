// RUN: heir-opt --annotate-level %s | FileCheck %s

module {
  func.func @simple(%arg0: !secret.secret<tensor<16xf32>>) -> !secret.secret<tensor<16xf32>> {
    // CHECK: secret.generic
    %0 = secret.generic(%arg0 : !secret.secret<tensor<16xf32>>) {
    ^body(%val: tensor<16xf32>):
      // CHECK: mgmt.modreduce
      // CHECK-SAME: {mgmt.level = 1 : index}
      %1 = mgmt.modreduce %val : tensor<16xf32>
      secret.yield %1 : tensor<16xf32>
    } -> !secret.secret<tensor<16xf32>>
    return %0 : !secret.secret<tensor<16xf32>>
  }
}
