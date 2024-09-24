// RUN: heir-opt --yosys-optimizer %s | FileCheck %s

module {
  // CHECK-LABEL: @capture
  // CHECK-SAME: %[[arr:.*]]: !secret.secret<memref<8xi8>>, %[[i:.*]]: index
  func.func @capture(%arr: !secret.secret<memref<8xi8>>, %i: index) -> !secret.secret<i8> {
    // CHECK-DAG: %[[v0:.*]] = secret.cast %[[arr]] : !secret.secret<memref<8xi8>> to !secret.secret<memref<64xi1>>
    // CHECK-DAG: %[[v1:.*]] = arith.index_cast %[[i]] : index to i3
    // CHECK: %[[alloc:.*]] = memref.alloc() : memref<3xi1>
    // CHECK: %[[v2:.*]] = secret.generic ins(%[[v0]], %[[alloc]] : !secret.secret<memref<64xi1>>, memref<3xi1>)
      // CHECK-COUNT-28: comb.truth_table
      // CHECK: secret.yield
    // CHECK: %[[v3:.*]] = secret.cast %[[v2]] : !secret.secret<memref<8xi1>> to !secret.secret<i8>
    // CHECK: return %[[v3]] : !secret.secret<i8>
    %0 = secret.generic ins(%arr: !secret.secret<memref<8xi8>>) {
        ^bb0(%ARR: memref<8xi8>):
            %1 = memref.load %ARR[%i] : memref<8xi8> // %i gets implicitly captured here
            %2 = arith.addi %1, %1 : i8 // arithmetic so yosys doesn't skip us
            secret.yield %2 : i8
    } -> (!secret.secret<i8>)
    return %0 : !secret.secret<i8>
  }
}
