// RUN: heir-opt --secret-insert-mgmt-ckks="bootstrap-waterline=2 level-budget=4" %s | FileCheck %s

// A waterline crossing whose remaining consumption cone terminates within the
// declared level budget needs no refresh at all: the budgeted modulus chain
// accommodates the few extra levels, and the value is consumed before the
// chain runs out.

// CHECK: func.func @skip_terminal_refresh
// CHECK-NOT: mgmt.bootstrap
// CHECK:      %[[m2:[^ ]+]] = mgmt.modreduce {{.*}} {mgmt.mgmt = #mgmt.mgmt<level = 0>}
// CHECK-NEXT: %[[s:[^ ]+]] = arith.addf %[[m2]], %[[m2]] {mgmt.mgmt = #mgmt.mgmt<level = 0>}
// CHECK-NOT: mgmt.bootstrap

module attributes {backend.lattigo, scheme.ckks, backend.config_override = {bootstrapLevelsConsumed = 0 : i32}} {
func.func @skip_terminal_refresh(%arg0: !secret.secret<tensor<16xf32>>) -> !secret.secret<tensor<16xf32>> {
  %0 = secret.generic(%arg0: !secret.secret<tensor<16xf32>>) {
  ^body(%input0: tensor<16xf32>):
    %1 = arith.addf %input0, %input0 : tensor<16xf32>
    %r0 = mgmt.modreduce %1 : tensor<16xf32>
    %2 = arith.addf %r0, %r0 : tensor<16xf32>
    %r1 = mgmt.modreduce %2 : tensor<16xf32>
    %3 = arith.addf %r1, %r1 : tensor<16xf32>
    // This modreduce crosses the waterline (2), but the remaining cone (one
    // add, then yield) fits inside the level budget (4), so no bootstrap.
    %r2 = mgmt.modreduce %3 : tensor<16xf32>
    %4 = arith.addf %r2, %r2 : tensor<16xf32>
    secret.yield %4 : tensor<16xf32>
  } -> !secret.secret<tensor<16xf32>>
  return %0 : !secret.secret<tensor<16xf32>>
}
}
