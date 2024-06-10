// RUN: heir-opt %s | FileCheck %s

// CHECK-LABEL: @test_arith_syntax
func.func @test_arith_syntax() {
  %zero = arith.constant 1 : i10
  %cmod = arith.constant 17 : i10
  %c_vec = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi10>
  %cmod_vec = arith.constant dense<17> : tensor<4xi10>

  // CHECK: arith_ext.barrett_reduce
  // CHECK: arith_ext.barrett_reduce
  %barrett = arith_ext.barrett_reduce %zero { modulus = 17 } : i10
  %barrett_vec = arith_ext.barrett_reduce %c_vec { modulus = 17 } : tensor<4xi10>

  // CHECK: arith_ext.subifge
  // CHECK: arith_ext.subifge
  %subifge = arith_ext.subifge %zero, %cmod : i10
  %subifge_vec = arith_ext.subifge %c_vec, %cmod_vec : tensor<4xi10>

  return
}
