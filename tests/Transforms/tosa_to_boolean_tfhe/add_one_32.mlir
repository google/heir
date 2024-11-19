// RUN: heir-opt --tosa-to-boolean-tfhe=entry-function=add_one %s | FileCheck %s

// While this is not a TOSA model, it should still lower through the pipeline.

module {
  // CHECK: @add_one([[sks:.*]]: !tfhe_rust.server_key, [[arg:.*]]: memref<8x!tfhe_rust.eui3>)
  // CHECK-NOT: comb
  // CHECK-NOT: arith.{{^constant}}
  // CHECK-COUNT-11: tfhe_rust.apply_lookup_table
  // CHECK: return
  func.func @add_one(%in: i32) -> (i32) {
    %1 = arith.constant 1 : i32
    %3 = arith.constant 6 : i32
    %2 = arith.addi %in, %1 : i32
    %4 = arith.muli %2, %3: i32
    return %4 : i32
  }
}
