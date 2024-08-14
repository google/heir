// RUN: heir-opt %s | FileCheck %s

// CHECK-LABEL: @test_arith_syntax
func.func @test_arith_syntax() {
  %zero = arith.constant 1 : i10
  %c4 = arith.constant 4 : i10
  %c5 = arith.constant 5 : i10
  %c6 = arith.constant 6 : i10
  %cmod = arith.constant 17 : i10
  %c_vec = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi10>
  %c_vec2 = arith.constant dense<[4, 3, 2, 1]> : tensor<4xi10>
  %c_vec3 = arith.constant dense<[1, 1, 1, 1]> : tensor<4xi10>
  %cmod_vec = arith.constant dense<17> : tensor<4xi10>

  // CHECK: mod_arith.add
  // CHECK: mod_arith.add
  %add = mod_arith.add %c5, %c6 { modulus = 17 } : i10
  %add_vec = mod_arith.add %c_vec, %c_vec2 { modulus = 17 } : tensor<4xi10>

  // CHECK: mod_arith.sub
  // CHECK: mod_arith.sub
  %sub = mod_arith.sub %c5, %c6 { modulus = 17 } : i10
  %sub_vec = mod_arith.sub %c_vec, %c_vec2 { modulus = 17 } : tensor<4xi10>

  // CHECK: mod_arith.mul
  // CHECK: mod_arith.mul
  %mul = mod_arith.mul %c5, %c6 { modulus = 17 } : i10
  %mul_vec = mod_arith.mul %c_vec, %c_vec2 { modulus = 17 } : tensor<4xi10>

  // CHECK: mod_arith.mac
  // CHECK: mod_arith.mac
  %mac = mod_arith.mac %c5, %c6, %c4 { modulus = 17 } : i10
  %mac_vec = mod_arith.mac %c_vec, %c_vec2, %c_vec3 { modulus = 17 } : tensor<4xi10>

  // CHECK: mod_arith.barrett_reduce
  // CHECK: mod_arith.barrett_reduce
  %barrett = mod_arith.barrett_reduce %zero { modulus = 17 } : i10
  %barrett_vec = mod_arith.barrett_reduce %c_vec { modulus = 17 } : tensor<4xi10>

  // CHECK: mod_arith.subifge
  // CHECK: mod_arith.subifge
  %subifge = mod_arith.subifge %zero, %cmod : i10
  %subifge_vec = mod_arith.subifge %c_vec, %cmod_vec : tensor<4xi10>

  return
}
