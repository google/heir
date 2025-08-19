// RUN: heir-opt --new-layout-propagation=ciphertext-size=1024 %s | FileCheck %s

// CHECK-DAG: [[layout:#[^ ]*]] = #tensor_ext.new_layout<domainSize=0, relation="(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 1023 >= 0)">

// Layouts should not be assigned to the cleartext function args
// CHECK: func @cmux
// CHECK-SAME: [[arg0:%[^:]*]]: i64,
// CHECK-SAME: [[arg1:%[^:]*]]: i64,
// CHECK-SAME: [[arg2:%[^:]*]]: !secret.secret<i1> {tensor_ext.layout = [[layout]]}
func.func @cmux(%arg0: i64, %arg1: i64, %arg2: !secret.secret<i1>) -> !secret.secret<i64> {
  %true = arith.constant true
  %0 = secret.generic(%arg2 : !secret.secret<i1>) {
  ^body(%input0: i1):
    %1 = arith.subi %true, %input0 : i1
    %2 = arith.extui %input0 : i1 to i64
    %3 = arith.muli %2, %arg0 : i64
    %4 = arith.extui %1 : i1 to i64
    %5 = arith.muli %4, %arg1 : i64
    %6 = arith.addi %3, %5 : i64
    secret.yield %6 : i64
  } -> !secret.secret<i64>
  return %0 : !secret.secret<i64>
}
