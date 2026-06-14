// RUN: heir-translate %s --emit-openfhe-emitc | FileCheck %s

module attributes {scheme.ckks} {
  // CHECK: CiphertextT test_affine_for(CryptoContextT [[v1:[a-zA-Z0-9_]+]], CiphertextT [[v2:[a-zA-Z0-9_]+]]) {
  // CHECK-NEXT: int64_t [[v3:[a-zA-Z0-9_]+]] = 1;
  // CHECK-NEXT: CiphertextT [[v4:[a-zA-Z0-9_]+]] = [[v1]].EvalRotate([[v2]], [[v3]]);
  // CHECK-NEXT: return [[v4]];
  // CHECK-NEXT: }
  emitc.func @test_affine_for(%arg0: !emitc.opaque<"CryptoContextT">, %arg1: !emitc.opaque<"CiphertextT">) -> !emitc.opaque<"CiphertextT"> {
    %0 = "emitc.constant"() <{value = #emitc.opaque<"1">}> : () -> !emitc.opaque<"int64_t">
    %1 = member_call_opaque %arg0 "EvalRotate"(%arg1, %0)  : !emitc.opaque<"CryptoContextT">, (!emitc.opaque<"CiphertextT">, !emitc.opaque<"int64_t">) -> !emitc.opaque<"CiphertextT">
    return %1 : !emitc.opaque<"CiphertextT">
  }
}
