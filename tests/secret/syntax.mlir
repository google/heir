// RUN: heir-opt %s > %t
// RUN: FileCheck %s < %t

// This simply tests for syntax.

module {
  // CHECK-LABEL: func @fooFunc
  func.func @fooFunc(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
    return %arg0 : !secret.secret<i32>
  }
}
