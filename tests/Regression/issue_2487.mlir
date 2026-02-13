// RUN: heir-opt --secret-to-mod-arith=modulus=17 %s | FileCheck %s

// CHECK: module
// CHECK: func.func @main
// CHECK-NOT: mgmt.modreduce
// CHECK: return
module {
  func.func @main(%arg0: !secret.secret<tensor<8xi8>>) -> !secret.secret<tensor<8xi8>> {
    %0 = secret.generic(%arg0: !secret.secret<tensor<8xi8>>) {
    ^bb0(%arg1: tensor<8xi8>):
      %1 = mgmt.modreduce %arg1 : tensor<8xi8>
      secret.yield %1 : tensor<8xi8>
    } -> !secret.secret<tensor<8xi8>>
    return %0 : !secret.secret<tensor<8xi8>>
  }
}
