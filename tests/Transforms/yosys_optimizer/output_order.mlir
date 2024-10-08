// RUN: heir-opt --yosys-optimizer --canonicalize %s | FileCheck %s

// This tests the output ordering of Yosys Optimizer by ensuring that doubling
// an input results in a final zero bit.

module {
  // CHECK-LABEL: @output_order
  func.func @output_order(%in: !secret.secret<i2>) -> (!secret.secret<i2>) {
    // CHECK: [[C0:%.*]] = arith.constant 0 : index
    // CHECK: [[FALSE:%.*]] = arith.constant false
    // CHECK: [[V1:%.*]] = secret.generic
    %1 = secret.generic
        ins(%in: !secret.secret<i2>) {
        ^bb0(%IN: i2) :
            // CHECK-NOT: arith.addi
            // CHECK: [[ALLOC:%.*]] = memref.alloc()
            // CHECK: memref.store [[FALSE]], [[ALLOC]][[[C0]]]
            %2 = arith.addi %IN, %IN : i2
            secret.yield %2 : i2
        } -> (!secret.secret<i2>)
    // CHECK: [[V2:%.*]] = secret.cast [[V1]]
    // CHECK-SAME:   !secret.secret<memref<2xi1>> to !secret.secret<i2>
    // CHECK: return [[V2]]
    return %1 : !secret.secret<i2>
  }
}
