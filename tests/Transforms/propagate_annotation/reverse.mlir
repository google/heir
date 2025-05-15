// RUN: heir-opt %s --propagate-annotation="attr-name=test.attr reverse=true" | FileCheck %s

// CHECK: @mul
// CHECK-SAME: (%[[arg0:.*]]: i16 {test.attr = 3
// CHECK-SAME: ) -> (i16 {test.attr = 3
func.func @mul(%arg0 : i16) -> (i16 {test.attr = 3}) {
  // CHECK: arith.muli
  // CHECK-SAME: {test.attr = 3
  %1 = arith.muli %arg0, %arg0 : i16
  // Only the first muli is used later, so the other two muli's don't get
  // propagated to.
  // CHECK-NOT: test.attr
  %2 = arith.muli %1, %1 : i16
  %3 = arith.muli %2, %1 : i16
  return %1 : i16
}
