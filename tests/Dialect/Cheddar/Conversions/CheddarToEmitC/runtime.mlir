// RUN: heir-opt --cheddar-emitc-boundary %s | FileCheck %s

// The `heir::` runtime helpers are emitted alongside the generated code, after
// the CHEDDAR prelude.

// CHECK: emitc.verbatim "using word = uint64_t;"
// CHECK-NEXT: emitc.verbatim "namespace heir {
// CHECK-SAME: getEncoder
// CHECK-SAME: keys.GetMultiplicationKey();

func.func @empty() {
  return
}
