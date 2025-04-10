// RUN: heir-opt --secret-insert-mgmt-ckks %s | FileCheck %s

// CHECK-NOT: mgmt.init
func.func @cleartext_arith(%0: index) -> index {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %22 = arith.remsi %0, %c32 : index
  %23 = arith.cmpi slt, %22, %c0 : index
  %26 = arith.addi %22, %22 : index
  %27 = arith.select %23, %26, %22 : index
  return %27 : index
}
