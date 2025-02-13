// RUN: heir-opt --mlir-to-bgv --bgv-to-lwe --lwe-to-lattigo --lattigo-alloc-to-inplace %s | FileCheck %s

// CHECK-LABEL: func.func @add
func.func @add(%arg0 : i16 {secret.secret}) -> i16 {
    // CHECK-COUNT-3: lattigo.bgv.add
    // CHECK-NOT: lattigo.bgv.add_new
    %0 = arith.addi %arg0, %arg0 : i16
    %1 = arith.addi %0, %0 : i16
    %2 = arith.addi %1, %1 : i16
    return %2 : i16
}
