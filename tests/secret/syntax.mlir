// RUN: heir-opt %s > %t
// RUN: FileCheck %s < %t

// This simply tests for syntax.

module {
  // CHECK-LABEL: func @fooFunc
  func.func @fooFunc(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
    return %arg0 : !secret.secret<i32>
  }

  // CHECK-LABEL: noInputs
  func.func @noInputs() -> !secret.secret<memref<1x16xi8>> {
    // CHECK: secret.generic
    %Z = secret.generic {
        %d = memref.alloc() {alignment = 64 : i64} : memref<1x16xi8>
        secret.yield %d : memref<1x16xi8>
      } -> (!secret.secret<memref<1x16xi8>>)
    func.return %Z : !secret.secret<memref<1x16xi8>>
  }
}
