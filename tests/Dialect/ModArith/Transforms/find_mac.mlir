// RUN: heir-opt --arith-to-mod-arith --mod-arith-to-mac %s | FileCheck %s --enable-var-scope

// CHECK-LABEL: @double_mac
// CHECK-SAME: (%[[ARG:.*]]: !Z32768_i17) -> [[T:.*]] {
func.func @double_mac(%arg0: i16) -> i16 {
  %c1 = arith.constant 1: i16
  %c2 = arith.constant 2 : i16
  %c3 = arith.constant 3: i16

  // CHECK: %[[MAC1:.*]] = mod_arith.mac %[[ARG]], %{{.*}}, %{{.*}} : [[T]]
  // CHECK: %[[ADD:.*]] = mod_arith.add %[[MAC1]], %{{.*}} : [[T]]
  // CHECK: %[[MAC2:.*]] = mod_arith.mac %[[ADD]], %{{.*}}, %{{.*}} : [[T]]
  // CHECK: return %[[MAC2]] : [[T]]

  %3 = arith.muli %arg0, %c2 : i16
  %4 = arith.addi %3, %c1 : i16
  %7 = arith.addi %4, %c3 : i16

  %5 = arith.muli %7, %c3 : i16
  %6 = arith.addi %c2, %5 : i16

  return %6 : i16
}
