// RUN: heir-opt --annotate-mgmt %s | FileCheck %s

func.func @main(%arg0: !secret.secret<tensor<8xi8>>) -> !secret.secret<tensor<8xi8>> {
  %b = secret.generic(%arg0: !secret.secret<tensor<8xi8>>) {
  ^body(%clear_a: tensor<8xi8>):
    %c = mgmt.modreduce %clear_a : tensor<8xi8>
    secret.yield %c : tensor<8xi8>
  // %b here should have level _annotation_ of 0 instead of 1.
  // CHECK: } -> (!secret.secret<tensor<8xi8>> {mgmt.mgmt = #mgmt.mgmt<level = 0>})
  } -> !secret.secret<tensor<8xi8>>
  func.return %b : !secret.secret<tensor<8xi8>>
}
