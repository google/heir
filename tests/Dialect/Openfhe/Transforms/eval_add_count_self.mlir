// RUN: heir-opt --annotate-module="backend=openfhe scheme=bfv" --mlir-to-bfv --scheme-to-openfhe %s | FileCheck %s

// CHECK: evalAddCount = 4
func.func @add_self(%arg0 : i16 {secret.secret}) -> i16 {
    %0 = arith.addi %arg0, %arg0 : i16
    %1 = arith.addi %0, %0 : i16
    return %1 : i16
}
