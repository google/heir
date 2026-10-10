// RUN: heir-opt --cheddar-emitc-boundary --cheddar-emitc-boundary %s | FileCheck %s

// Running the boundary pass twice emits the prelude only once.

// CHECK-COUNT-1: emitc.include "core/Context.h"
// CHECK-NOT: emitc.include "core/Context.h"
// CHECK-COUNT-1: emitc.verbatim "using namespace cheddar;"
// CHECK-NOT: emitc.verbatim "using namespace cheddar;"
// CHECK: func.func @empty

func.func @empty() {
  return
}
