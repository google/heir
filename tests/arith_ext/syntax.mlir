// RUN: heir-opt %s | FileCheck %s

// CHECK-LABEL: @test_arith_syntax
func.func @test_arith_syntax() {
  %zero = arith.constant 1 : i10
  %c5 = arith.constant 5 : i10
  %c6 = arith.constant 6 : i10
  %cmod = arith.constant 17 : i10
  %c_vec = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi10>
  %c_vec2 = arith.constant dense<[4,3,2,1]> : tensor<4xi10>
  %cmod_vec = arith.constant dense<17> : tensor<4xi10>

  // CHECK: arith_ext.add
  // CHECK: arith_ext.add
  %add = arith_ext.add %c5, %c6 { modulus = 17 } : i10
  %add_vec = arith_ext.add %c_vec, %c_vec2 { modulus = 17 } : tensor<4xi10>

  // CHECK: arith_ext.sub
  // CHECK: arith_ext.sub
  %sub = arith_ext.sub %c5, %c6 { modulus = 17 } : i10
  %sub_vec = arith_ext.sub %c_vec, %c_vec2 { modulus = 17 } : tensor<4xi10>

  // CHECK: arith_ext.mul
  // CHECK: arith_ext.mul
  %mul = arith_ext.mul %c5, %c6 { modulus = 17 } : i10
  %mul_vec = arith_ext.mul %c_vec, %c_vec2 { modulus = 17 } : tensor<4xi10>

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
