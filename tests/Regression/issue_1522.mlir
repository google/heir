// RUN: heir-opt %s --mlir-to-secret-arithmetic --secret-insert-mgmt-bgv \
// RUN:  --generate-param-bgv --openfhe-count-add-and-key-switch | FileCheck %s

// CHECK-LABEL: @foo
func.func @foo(%a : i16 {secret.secret})  -> i32 {
    %aa = arith.extui %a : i16 to i32
    return %aa : i32
}
