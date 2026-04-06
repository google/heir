// RUN: heir-opt --annotate-mgmt %s | FileCheck %s

func.func @main(%arg0: !secret.secret<tensor<8xi8>>) -> !secret.secret<tensor<8xi8>> {
  // CHECK: secret.generic
  // CHECK-SAME: level = 2
  %b = secret.generic(%arg0: !secret.secret<tensor<8xi8>>) {
  ^body(%clear_a: tensor<8xi8>):
    %c = mgmt.level_reduce %clear_a { levelToDrop = 2 }: tensor<8xi8>
    secret.yield %c : tensor<8xi8>
  // CHECK: } -> (!secret.secret<tensor<8xi8>> {mgmt.mgmt = #mgmt.mgmt<level = 0>})
  } -> !secret.secret<tensor<8xi8>>
  func.return %b : !secret.secret<tensor<8xi8>>
}
