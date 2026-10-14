// RUN: heir-opt --cheddar-emitc-boundary %s | FileCheck %s

// The `heir::` runtime helper is emitted alongside the generated code, after
// the CHEDDAR prelude. The multiplication key is a direct EvkMap call and
// needs no helper.

// CHECK: emitc.verbatim "using word = uint64_t;"
// CHECK-NEXT: emitc.verbatim "namespace heir {
// CHECK-SAME: getEncoder
// CHECK-NOT: multiplicationKey

func.func @empty() {
  return
}
