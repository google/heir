// RUN: heir-opt --tosa-to-boolean-tfhe %s | FileCheck %s

// While this is not a TOSA model, it should still lower through the pipeline.

module {
  // CHECK: @main([[sks:.*]]: !tfhe_rust.server_key, [[arg:.*]]: memref<8x!tfhe_rust.eui3>)
  // CHECK-NOT: comb
  // CHECK-NOT: arith.{{^constant}}
  // CHECK-COUNT-15: tfhe_rust.apply_lookup_table
  func.func @main(%in: i8) -> (i8) {
    %1 = arith.constant 1 : i8
    %2 = arith.addi %in, %1 : i8
    return %2 : i8
  }
}
