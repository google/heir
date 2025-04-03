// RUN: heir-opt --mlir-to-secret-arithmetic --secret-insert-mgmt-bgv --optimize-relinearization %s | FileCheck %s

// CHECK: func.func @repro
// CHECK-COUNT-1: mgmt.relinearize
// CHECK-NOT: mgmt.relinearize
func.func @repro(%x: i16 {secret.secret}, %y: i16 {secret.secret}, %p: i16) -> (i16) {
    %xx = arith.muli %x, %x : i16
    %yy = arith.muli %y, %y : i16
    %0 = arith.addi %xx, %yy : i16
    %xp = arith.muli %x, %p : i16
    %1 = arith.addi %xp, %0 : i16
    func.return %1 : i16
}

// CHECK: func.func @repro2
// CHECK-COUNT-1: mgmt.relinearize
// CHECK-NOT: mgmt.relinearize
func.func @repro2(%x: i16 {secret.secret}, %y: i16 {secret.secret}, %p: i16) -> (i16) {
    %xx = arith.muli %x, %x : i16
    %yy = arith.muli %y, %y : i16
    %0 = arith.addi %xx, %yy : i16
    %xp = arith.muli %x, %p : i16
    %1 = arith.addi %0, %xp : i16
    func.return %1 : i16
}
