// RUN: heir-opt %s | FileCheck %s

// CHECK-LABEL: func @fooFunc
func.func @fooFunc(%a: !heir.ciphertext, %b: !heir.ciphertext) -> (!heir.ciphertext) {
// CHECK: %0 = heir.add(%arg0, %arg1) : (!heir.ciphertext, !heir.ciphertext) -> !heir.ciphertext
  %result = heir.add(%a, %b) : (!heir.ciphertext, !heir.ciphertext) -> !heir.ciphertext
// CHECK: return %0 : !heir.ciphertext
  func.return %result : !heir.ciphertext
}
