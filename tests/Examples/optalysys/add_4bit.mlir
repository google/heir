// RUN: heir-opt --tosa-to-boolean-optalysys %s | FileCheck %s

// CHECK-LABEL: main
// CHECK-COUNT-4: optalysys.device_multi_cmux_1024_pbs_b
// CHECK: return
func.func @main(%arg0 : i4) -> i4 {
      %c1 = arith.constant 1 : i4
      %0 = arith.addi %arg0, %c1 : i4
      return %0 : i4
}
