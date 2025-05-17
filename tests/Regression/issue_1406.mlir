// RUN: heir-opt %s "--annotate-module=backend=openfhe scheme=bgv" "--mlir-to-bgv=ciphertext-degree=8192" | FileCheck %s

// CHECK-NOT: !secret.secret<i16>
func.func @add(%arg0 : i16 {secret.secret}, %arg1 : i16 {secret.secret}) -> i16 {
    %0 = arith.addi %arg0, %arg0 : i16
    return %0 : i16
}
