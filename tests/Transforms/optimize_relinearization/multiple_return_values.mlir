// RUN: heir-opt --mlir-to-secret-arithmetic=ciphertext-degree=8 --secret-insert-mgmt-bgv --optimize-relinearization %s | FileCheck %s

// CHECK: func.func @repro
func.func @repro(%x: i16 {secret.secret}, %y: i16 {secret.secret}) -> (i16, i16) {
  %0 = arith.addi %x, %y : i16
  // CHECK: secret.yield %[[v1:[^ ,]*]], %[[v2:[^ ]*]] : tensor<8xi16>, tensor<8xi16>
  return %0, %0: i16, i16
}
