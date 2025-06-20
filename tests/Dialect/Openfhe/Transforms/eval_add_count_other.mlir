// RUN: heir-opt --annotate-module="backend=openfhe scheme=bfv" --mlir-to-bfv --scheme-to-openfhe %s | FileCheck %s

// CHECK: evalAddCount = 3
func.func @add_other(%arg0 : i16 {secret.secret}, %arg1 : i16 {secret.secret}) -> i16 {
    %0 = arith.addi %arg0, %arg1 : i16
    %1 = arith.addi %0, %arg1 : i16
    return %1 : i16
}
