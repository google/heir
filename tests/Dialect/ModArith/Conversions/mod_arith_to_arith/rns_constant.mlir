// RUN: heir-opt --mod-arith-to-arith %s | FileCheck %s

!Z0 = !mod_arith.int<17 : i32>
!Z1 = !mod_arith.int<97 : i32>
!rns = !rns.rns<!Z0, !Z1>

// CHECK: @test_lower_rns_constant
// CHECK-SAME: () -> tensor<2xi32>
func.func @test_lower_rns_constant() -> !rns {
  // CHECK: %[[CST:.*]] = arith.constant dense<[1, 2]> : tensor<2xi32>
  %0 = rns.constant <[#mod_arith.value<1 : !Z0> : !Z0, #mod_arith.value<2 : !Z1> : !Z1]> : !rns
  // CHECK: return %[[CST]]
  return %0 : !rns
}

// CHECK: @test_lower_splat_rns_constant
// CHECK-SAME: () -> tensor<4x2xi32>
func.func @test_lower_splat_rns_constant() -> tensor<4x!rns> {
  // CHECK: %[[CST:.*]] = arith.constant dense<[1, 2]> : tensor<2xi32>
  %0 = rns.constant <[#mod_arith.value<1 : !Z0> : !Z0, #mod_arith.value<2 : !Z1> : !Z1]> : !rns
  // CHECK: %[[BROADCAST:.*]] = linalg.broadcast ins(%[[CST]] : tensor<2xi32>)
  %1 = tensor.splat %0 : tensor<4x!rns>
  // CHECK: return %[[BROADCAST]]
  return %1 : tensor<4x!rns>
}
