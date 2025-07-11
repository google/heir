// RUN: heir-opt --mlir-to-cggi --scheme-to-tfhe-rs %s | FileCheck %s

module {
  // CHECK: @add_one([[sks:.*]]: !tfhe_rust.server_key, [[arg:.*]]: tensor<8x!tfhe_rust.eui3>)
  // CHECK-NOT: comb
  // CHECK-NOT: arith.{{^constant}}
  // CHECK-COUNT-11: tfhe_rust.apply_lookup_table
  // CHECK: return
  func.func @add_one(%in: i8 {secret.secret}) -> (i8) {
    %1 = arith.constant 1 : i8
    %2 = arith.addi %in, %1 : i8
    return %2 : i8
  }
}
