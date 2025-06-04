// RUN: heir-opt %s --split-input-file --propagate-annotation=attr-name=test.attr | FileCheck %s

func.func @mul(%arg0 : i16 {secret.secret}) -> i16 {
  // CHECK-COUNT-3: test.attr = 3
  %1 = arith.muli %arg0, %arg0 {test.attr = 3} : i16
  %2 = arith.muli %1, %1 : i16
  %3 = arith.muli %2, %1 : i16
  return %1 : i16
}

// -----

// 3 because one is an argAttr, one is a resultAttr, and one is on the first
// muli op.
// CHECK-COUNT-3: test.attr = 2
func.func @mul(%arg0 : i16 {secret.secret, test.attr = 2}) -> i16 {
  %1 = arith.muli %arg0, %arg0 : i16
  // CHECK-COUNT-2: test.attr = 1
  %2 = arith.muli %1, %1 {test.attr = 1} : i16
  %3 = arith.muli %2, %1 : i16
  return %1 : i16
}
