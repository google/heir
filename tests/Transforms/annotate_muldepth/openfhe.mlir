// RUN: heir-opt --annotate-muldepth --mlir-print-local-scope %s | FileCheck %s

// Tests other secret-like type by interface

// CHECK: func.func @doctest
// CHECK-SAME: !openfhe.ciphertext {secret.mul_depth = 0
// CHECK-NEXT: openfhe.mul
// CHECK-SAME: secret.mul_depth = 1
// CHECK-NEXT: openfhe.mul_plain
// CHECK-SAME: secret.mul_depth = 1
// CHECK-NEXT: openfhe.mul
// CHECK-SAME: secret.mul_depth = 2
// CHECK-NEXT: openfhe.mul
// CHECK-SAME: secret.mul_depth = 3
// CHECK-NEXT: return

!ct = !openfhe.ciphertext
!pt = !openfhe.plaintext
!cc = !openfhe.crypto_context

func.func @doctest(%cc: !cc, %secret_val: !ct, %public_val: !pt) -> !ct {
  %2 = openfhe.mul %cc, %secret_val, %secret_val : (!cc, !ct, !ct) -> !ct
  %3 = openfhe.mul_plain %cc, %secret_val, %public_val : (!cc, !ct, !pt) -> !ct
  %4 = openfhe.mul %cc, %2, %3 : (!cc, !ct, !ct) -> !ct
  %5 = openfhe.mul %cc, %4, %3 : (!cc, !ct, !ct) -> !ct
  return %5 : !ct
}
