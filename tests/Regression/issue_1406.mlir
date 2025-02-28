// RUN: heir-opt %s --mlir-to-secret-arithmetic --secret-insert-mgmt-bgv --generate-param-bgv --secret-distribute-generic --secret-to-bgv | FileCheck %s

// CHECK-NOT: !secret.secret<i16>
func.func @add(%arg0 : i16 {secret.secret}, %arg1 : i16 {secret.secret}) -> i16 {
    %0 = arith.addi %arg0, %arg0 : i16
    return %0 : i16
}
