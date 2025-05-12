// RUN: heir-opt --canonicalize %s | FileCheck %s

// CHECK: func @func
// CHECK-SAME: (%arg0: !secret.secret<i16>) -> i16
func.func @func(%arg0: !secret.secret<i16>) -> !secret.secret<i16> {
    // CHECK-NOT: secret.generic
    %0 = secret.generic(%arg0 : !secret.secret<i16>) {
    ^bb0(%arg1: i16):
        %c1_i16 = arith.constant 1 : i16
        secret.yield %c1_i16 : i16
    } -> !secret.secret<i16>
    //CHECK: return %[[S:.*]] : i16
    return %0 : !secret.secret<i16>
}
