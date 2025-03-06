// RUN: heir-opt %s --arith-to-mod-arith=modulus=65537 | FileCheck %s

// CHECK: !Z65537_i64 = !mod_arith.int<65537 : i64>

// CHECK-LABEL: @add
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x!Z65537_i64> {secret.secret}) -> tensor<8x!Z65537_i64> {
func.func @add(%arg0 : tensor<8xi16> {secret.secret}) -> tensor<8xi16> {
    %0 = arith.addi %arg0, %arg0 : tensor<8xi16>
    return %0 : tensor<8xi16>
}
