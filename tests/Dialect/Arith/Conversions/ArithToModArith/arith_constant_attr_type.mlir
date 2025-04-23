// RUN: heir-opt --arith-to-mod-arith=modulus=65536 --mlir-print-op-generic %s | FileCheck %s

// CHECK: [[mod:.*]] = !mod_arith.int<65536 : i64>

// CHECK: test_arith_constant
func.func @test_arith_constant() -> i16 {
  // CHECK: mod_arith.constant
  // This test ensures that the attribute given to the constant op matches the
  // storage type of the result type. Otherwise downstream type conversion will
  // have type materialization problems.
  // CHECK-SAME: <{value = 17 : i64}>
  %c17 = arith.constant 17 : i16
  return %c17 : i16
}


// CHECK: test_arith_constant_dense
func.func @test_arith_constant_dense() -> tensor<2xi16> {
  // CHECK: mod_arith.constant
  // CHECK-SAME: <{value = dense<2> : tensor<2xi64>}>
  %c2 = arith.constant dense<[2, 2]> : tensor<2xi16>
  return %c2 : tensor<2xi16>
}
