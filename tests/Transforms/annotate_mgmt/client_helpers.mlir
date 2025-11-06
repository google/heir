// RUN: heir-opt --annotate-mgmt %s | FileCheck %s

// CHECK: @main
func.func @main(%arg0: !secret.secret<i16> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 0>}) -> (!secret.secret<i16> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 0>}) {
  return %arg0 : !secret.secret<i16>
}

// CHECK: @encrypt_helper
// CHECK-SAME: (%[[arg0:.*]]: i16) -> (!secret.secret<i16> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 0>}) attributes
func.func @encrypt_helper(%arg0: i16) -> !secret.secret<i16> attributes {client.enc_func = {func_name = "main", index = 0 : i64}} {
  // CHECK: secret.conceal
  // CHECK-SAME: {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 0>
  %0 = secret.conceal %arg0 : i16 -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}
// CHECK: @decrypt_helper
// CHECK-SAME: (%[[arg0:.*]]: !secret.secret<i16> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 0>
func.func @decrypt_helper(%arg0: !secret.secret<i16>) -> i16 attributes {client.dec_func = {func_name = "main", index = 0 : i64}} {
  %0 = secret.reveal %arg0 : !secret.secret<i16> -> i16
  return %0 : i16
}
