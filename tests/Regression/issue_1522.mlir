// RUN: heir-opt %s "--annotate-module=backend=openfhe scheme=bgv" "--mlir-to-bgv=ciphertext-degree=8192" --scheme-to-openfhe | FileCheck %s

// CHECK: @foo
func.func @foo(%a : i16 {secret.secret})  -> i32 {
    %aa = arith.extui %a : i16 to i32
    return %aa : i32
}
