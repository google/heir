// RUN: heir-translate %s --emit-openfhe-emitc | FileCheck %s

module attributes {scheme.ckks} {
  // CHECK: int32_t if_scalar(CryptoContextT [[v1:[a-zA-Z0-9_]+]], int32_t [[v2:[a-zA-Z0-9_]+]], PrivateKeyT [[v3:[a-zA-Z0-9_]+]]) {
  // CHECK-NEXT: int32_t [[v4:[a-zA-Z0-9_]+]] = 6;
  // CHECK-NEXT: bool [[v5:[a-zA-Z0-9_]+]] = [[v2]] >= [[v4]];
  // CHECK-NEXT: int32_t [[v6:[a-zA-Z0-9_]+]];
  // CHECK-NEXT: if ([[v5]]) {
  // CHECK-NEXT: int32_t [[v7:[a-zA-Z0-9_]+]] = [[v2]] % [[v4]];
  // CHECK-NEXT: [[v6]] = [[v7]];
  // CHECK-NEXT: } else {
  // CHECK-NEXT: [[v6]] = [[v2]];
  // CHECK-NEXT: }
  // CHECK-NEXT: int32_t [[v8:[a-zA-Z0-9_]+]] = [[v6]];
  // CHECK-NEXT: return [[v8]];
  // CHECK-NEXT: }
  emitc.func @if_scalar(%arg0: !emitc.opaque<"CryptoContextT">, %arg1: i32, %arg2: !emitc.opaque<"PrivateKeyT">) -> i32 {
    %0 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %1 = cmp ge, %arg1, %0 : (i32, i32) -> i1
    %2 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
    if %1 {
      %4 = rem %arg1, %0 : (i32, i32) -> i32
      assign %4 : i32 to %2 : <i32>
    } else {
      assign %arg1 : i32 to %2 : <i32>
    }
    %3 = load %2 : <i32>
    return %3 : i32
  }
}
