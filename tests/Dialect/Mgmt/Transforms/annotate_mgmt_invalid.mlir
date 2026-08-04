// RUN: heir-opt --annotate-mgmt="level-budget=0" -verify-diagnostics %s

func.func @main(%arg0: !secret.secret<tensor<8xi8>>) -> !secret.secret<tensor<8xi8>> {
  %b = secret.generic(%arg0: !secret.secret<tensor<8xi8>>) { // expected-error {{value has invalid level}}
  ^body(%clear_a: tensor<8xi8>):
    %c = mgmt.modreduce %clear_a : tensor<8xi8> // expected-error {{value has invalid level}}
    secret.yield %c : tensor<8xi8>
  } -> !secret.secret<tensor<8xi8>>
  func.return %b : !secret.secret<tensor<8xi8>>
}
