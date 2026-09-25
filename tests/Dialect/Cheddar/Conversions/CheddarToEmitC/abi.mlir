// RUN: heir-opt --convert-to-emitc=filter-dialects=cheddar --cheddar-emitc-boundary --reconcile-unrealized-casts %s | FileCheck %s

// Payload buffer arguments become C++ references; calls stay structured.

// CHECK: func.func @abi_inner(
// CHECK-SAME: !emitc.array<1x!emitc.opaque<"const Ciphertext<word>">>
// CHECK-SAME: !emitc.array<1x!emitc.opaque<"Ciphertext<word>">>
func.func @abi_inner(
    %input: memref<1x!cheddar.ciphertext>,
    %output: memref<1x!cheddar.ciphertext> {bufferize.result}) {
  return
}

// CHECK: func.func @abi_outer(
// CHECK-SAME: !emitc.array<1x!emitc.opaque<"const Ciphertext<word>">>
// CHECK-SAME: !emitc.array<1x!emitc.opaque<"Ciphertext<word>">>
// CHECK: emitc.call_opaque "abi_inner"
func.func @abi_outer(
    %input: memref<1x!cheddar.ciphertext>,
    %output: memref<1x!cheddar.ciphertext>) {
  func.call @abi_inner(%input, %output)
      : (memref<1x!cheddar.ciphertext>, memref<1x!cheddar.ciphertext>) -> ()
  return
}

// Mutability propagates to a fixed point through more than one call edge.
// CHECK: func.func @abi_outermost(
// CHECK-SAME: !emitc.array<1x!emitc.opaque<"const Ciphertext<word>">>
// CHECK-SAME: !emitc.array<1x!emitc.opaque<"Ciphertext<word>">>
// CHECK: emitc.call_opaque "abi_outer"
func.func @abi_outermost(
    %input: memref<1x!cheddar.ciphertext>,
    %output: memref<1x!cheddar.ciphertext>) {
  func.call @abi_outer(%input, %output)
      : (memref<1x!cheddar.ciphertext>, memref<1x!cheddar.ciphertext>) -> ()
  return
}

// Rewriting a call to a refified function preserves ordinary scalar results.
// CHECK: func.func @abi_result_inner
func.func @abi_result_inner(
    %value: i32,
    %output: memref<1x!cheddar.ciphertext> {bufferize.result}) -> i32 {
  return %value : i32
}

// CHECK: func.func @abi_result_outer
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = call @abi_result_inner
// CHECK: return %[[RESULT]] : i32
func.func @abi_result_outer(
    %value: i32,
    %output: memref<1x!cheddar.ciphertext>) -> i32 {
  %result = func.call @abi_result_inner(%value, %output)
      : (i32, memref<1x!cheddar.ciphertext>) -> i32
  return %result : i32
}
