// RUN: heir-opt --secret-insert-mgmt-ckks --split-input-file %s | FileCheck %s

// CHECK: func.func private @extract_plaintext(f32) -> f32
// CHECK: func.func @call_plaintext(%[[arg0:.*]]: !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 0>}) -> (!secret.secret<f32>  {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 0>})
// CHECK-NEXT:  %[[v0:.*]] = secret.generic(%[[arg0]]: !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 0>}) {
// CHECK-NEXT:     ^body(%[[input0:.*]]: f32)
// CHECK-NEXT:       %[[v1:.*]] = func.call @extract_plaintext(%[[input0]]) {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 0>} : (f32) -> f32
// CHECK-NEXT:       secret.yield %[[v1]] : f32
// CHECK-NEXT:  } -> (!secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 0>})
// CHECK-NEXT:  return %[[v0]] : !secret.secret<f32>
module {
  func.func private @extract_plaintext(f32) -> f32
  func.func @call_plaintext(%arg0: !secret.secret<f32>) -> !secret.secret<f32> {
    %0 = secret.generic(%arg0 : !secret.secret<f32>) {
    ^body(%input0: f32):
      %1 = func.call @extract_plaintext(%input0) : (f32) -> f32
      secret.yield %1 : f32
    } -> !secret.secret<f32>
    return %0 : !secret.secret<f32>
  }
}

// -----

// CHECK: func.func private @external_secret(!secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 0>}) -> !secret.secret<f32>
// CHECK: func.func @call_secret(%[[arg0:.*]]: !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 0>}) -> (!secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 0>})
// CHECK-NEXT:  %[[v0:.*]] = call @external_secret(%[[arg0]])
// CHECK-NEXT:  return %[[v0]]
module {
  func.func private @external_secret(!secret.secret<f32>) -> !secret.secret<f32>
  func.func @call_secret(%arg0: !secret.secret<f32>) -> !secret.secret<f32> {
    %0 = func.call @external_secret(%arg0) : (!secret.secret<f32>) -> !secret.secret<f32>
    return %0 : !secret.secret<f32>
  }
}
