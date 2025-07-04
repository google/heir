// RUN: heir-opt --mlir-print-local-scope %s | FileCheck %s

!Zp = !mod_arith.int<17 : i10>
!Zp2 = !mod_arith.int<257 : i10>
!Zp3 = !mod_arith.int<509 : i10>
!Zp_vec = tensor<4x!Zp>

!rns = !rns.rns<!Zp, !Zp2, !Zp3>
!rns_vec = tensor<5x6x!rns>
!int_vec = tensor<5x6x3xi10>
!Zp_large = !mod_arith.int<2223821 : i32>

// CHECK: @test_arith_syntax
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

  // CHECK: mod_arith.constant 12 : !mod_arith.int<17 : i10>
  %const123 = mod_arith.constant 12 : !Zp
  // CHECK: mod_arith.constant 0 : !mod_arith.int<17 : i10>
  %constZero = mod_arith.constant 0 : !Zp
  // CHECK: mod_arith.constant -1 : !mod_arith.int<17 : i10>
  %constNegative = mod_arith.constant -1 : !Zp
  // CHECK: mod_arith.constant dense<[1, 2, 3, 4]> : tensor<4x!mod_arith.int<17 : i10>>
  %constdense = mod_arith.constant dense<[1, 2, 3, 4]> : !Zp_vec
  // CHECK: mod_arith.constant dense<[0, 2, 3, 4]> : tensor<4x!mod_arith.int<17 : i10>>
  %constdenseZero = mod_arith.constant dense<[0, 2, 3, 4]> : !Zp_vec
  // CHECK: mod_arith.constant dense<[-1, -2, -3, -4]> : tensor<4x!mod_arith.int<17 : i10>>
  %constdenseNegative = mod_arith.constant dense<[-1, -2, -3, -4]> : !Zp_vec

  // CHECK-COUNT-6: mod_arith.encapsulate
  %e4 = mod_arith.encapsulate %c4 : i10 -> !Zp
  %e5 = mod_arith.encapsulate %c5 : i10 -> !Zp
  %e6 = mod_arith.encapsulate %c6 : i10 -> !Zp
  %e_vec = mod_arith.encapsulate %c_vec : tensor<4xi10> -> !Zp_vec
  %e_vec2 = mod_arith.encapsulate %c_vec2 : tensor<4xi10> -> !Zp_vec
  %e_vec3 = mod_arith.encapsulate %c_vec3 : tensor<4xi10> -> !Zp_vec

  // CHECK-COUNT-6: mod_arith.reduce
  %m4 = mod_arith.reduce %e4 : !Zp
  %m5 = mod_arith.reduce %e5 : !Zp
  %m6 = mod_arith.reduce %e6 : !Zp
  %m_vec = mod_arith.reduce %e_vec : !Zp_vec
  %m_vec2 = mod_arith.reduce %e_vec2 : !Zp_vec
  %m_vec3 = mod_arith.reduce %e_vec3 : !Zp_vec

  // CHECK: mod_arith.extract
  %extract = mod_arith.extract %m4 : !Zp -> i10
  %extract_vec = mod_arith.extract %m_vec : !Zp_vec -> tensor<4xi10>

  // CHECK: mod_arith.add
  // CHECK: mod_arith.add
  %add = mod_arith.add %m5, %m6 : !Zp
  %add_vec = mod_arith.add %m_vec, %m_vec2 : !Zp_vec

  // CHECK: mod_arith.sub
  // CHECK: mod_arith.sub
  %sub = mod_arith.sub %m5, %m6 : !Zp
  %sub_vec = mod_arith.sub %m_vec, %m_vec2 : !Zp_vec

  // CHECK: mod_arith.mul
  // CHECK: mod_arith.mul
  %mul = mod_arith.mul %m5, %m6 : !Zp
  %mul_vec = mod_arith.mul %m_vec, %m_vec2 : !Zp_vec

  // CHECK: mod_arith.mac
  // CHECK: mod_arith.mac
  %mac = mod_arith.mac %m5, %m6, %m4 : !Zp
  %mac_vec = mod_arith.mac %m_vec, %m_vec2, %m_vec3 : !Zp_vec

  // CHECK: mod_arith.barrett_reduce
  // CHECK: mod_arith.barrett_reduce
  %barrett = mod_arith.barrett_reduce %zero { modulus = 17 } : i10
  %barrett_vec = mod_arith.barrett_reduce %c_vec { modulus = 17 } : tensor<4xi10>

  // CHECK: mod_arith.subifge
  // CHECK: mod_arith.subifge
  %subifge = mod_arith.subifge %zero, %cmod : i10
  %subifge_vec = mod_arith.subifge %c_vec, %cmod_vec : tensor<4xi10>

  // CHECK: mod_arith.mod_switch
  // CHECK: mod_arith.mod_switch
  // CHECK: mod_arith.mod_switch
  %mod_switch_smaller = mod_arith.mod_switch %e6 : !Zp to !mod_arith.int<7 : i4>
  %mod_switch_same_width = mod_arith.mod_switch %e6 : !Zp to !mod_arith.int<31 : i10>
  %mod_switch_larger = mod_arith.mod_switch %e6 : !Zp to !mod_arith.int<18446744069414584321 : i65>

  return
}

// CHECK: @test_rns_syntax
func.func @test_rns_syntax(%arg0: !rns, %arg1: !rns) {
  // CHECK: mod_arith.add
  %add = mod_arith.add %arg0, %arg1 : !rns

  // CHECK: mod_arith.mod_switch
  // CHECK: mod_arith.mod_switch
  %interpolate = mod_arith.mod_switch %add : !rns to !Zp_large
  %decompose = mod_arith.mod_switch %interpolate : !Zp_large to !rns

  return
}

// CHECK: @test_rns_vec_syntax
func.func @test_rns_vec_syntax(%arg0: !rns_vec, %arg1: !rns_vec) {
  // CHECK: mod_arith.add
  %add = mod_arith.add %arg0, %arg1 : !rns_vec
  
  // CHECK: mod_arith.extract
  // CHECK: mod_arith.encapsulate
  %extract = mod_arith.extract %add : !rns_vec -> !int_vec
  %encapsulate = mod_arith.encapsulate %extract : !int_vec -> !rns_vec

  return
}
