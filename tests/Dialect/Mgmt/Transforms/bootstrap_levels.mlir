// RUN: heir-opt --annotate-module="backend=openfhe" --annotate-mgmt %s | FileCheck %s --check-prefix=CHECK-OPENFHE
// RUN: heir-opt --annotate-module="backend=lattigo" --annotate-mgmt %s | FileCheck %s --check-prefix=CHECK-LATTIGO

func.func @main(%arg0: !secret.secret<tensor<8xi8>>) -> !secret.secret<tensor<8xi8>> {
  %b = secret.generic(%arg0: !secret.secret<tensor<8xi8>>) {
  ^body(%clear_a: tensor<8xi8>):
    %c = mgmt.bootstrap %clear_a : tensor<8xi8>
    secret.yield %c : tensor<8xi8>
  } -> !secret.secret<tensor<8xi8>>
  func.return %b : !secret.secret<tensor<8xi8>>
}

// CHECK-OPENFHE: func.func @main(%arg0: !secret.secret<tensor<8xi8>> {mgmt.mgmt = #mgmt.mgmt<level = 3>})
// CHECK-OPENFHE: mgmt.bootstrap
// CHECK-OPENFHE-SAME: {mgmt.mgmt = #mgmt.mgmt<level = 0>{{.*}}}

// CHECK-LATTIGO: func.func @main(%arg0: !secret.secret<tensor<8xi8>> {mgmt.mgmt = #mgmt.mgmt<level = 1>})
// CHECK-LATTIGO: mgmt.bootstrap
// CHECK-LATTIGO-SAME: {mgmt.mgmt = #mgmt.mgmt<level = 0>{{.*}}}
