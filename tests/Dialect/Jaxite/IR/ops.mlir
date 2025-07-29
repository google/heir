// RUN: heir-opt %s | FileCheck %s

// This simply tests for syntax.

!bsks = !jaxite.server_key_set
!params = !jaxite.params

#key = #lwe.key<slot_index = 0>
#preserve_overflow = #lwe.preserve_overflow<>
#app_data = #lwe.application_data<message_type = i1, overflow = #preserve_overflow>
#poly = #polynomial.int_polynomial<x>
#pspace = #lwe.plaintext_space<ring = #polynomial.ring<coefficientType = i3, polynomialModulus = #poly>, encoding = #lwe.constant_coefficient_encoding<scaling_factor = 268435456>>
#cspace = #lwe.ciphertext_space<ring = #polynomial.ring<coefficientType = i32, polynomialModulus = #poly>, encryption_type = msb, size = 742>
!ciphertext = !lwe.lwe_ciphertext<application_data = #app_data, plaintext_space = #pspace, ciphertext_space = #cspace, key = #key>

module {
  // CHECK: func @test_create_trivial_bool
  func.func @test_create_trivial_bool(%params : !params) {
    %0 = arith.constant 1 : i1
    %1 = arith.constant 0 : i1

    %e1 = jaxite.constant %0, %params : (i1, !params) -> !ciphertext
    %e2 = jaxite.constant %1, %params : (i1, !params) -> !ciphertext
    return
  }

  // CHECK: func @test_lut3
  func.func @test_lut3(%bsks : !bsks, %params : !params) {
    %0 = arith.constant 1 : i1
    %1 = arith.constant 1 : i1
    %2 = arith.constant 1 : i1
    %truth_table = arith.constant 8 : i8

    %e1 = jaxite.constant %0, %params : (i1, !params) -> !ciphertext
    %e2 = jaxite.constant %1, %params : (i1, !params) -> !ciphertext
    %e3 = jaxite.constant %2, %params : (i1, !params) -> !ciphertext

    %out = jaxite.lut3 %e1, %e2, %e3, %truth_table, %bsks, %params : (!ciphertext, !ciphertext, !ciphertext, i8, !bsks, !params) -> !ciphertext
    return
  }

  // CHECK: func @test_packed_lut3
  func.func @test_packed_lut3(%bsks : !bsks, %params : !params) {
    %0 = arith.constant 1 : i1
    %1 = arith.constant 1 : i1
    %2 = arith.constant 1 : i1
    %truth_table = arith.constant 8 : i8

    %e1 = jaxite.constant %0, %params : (i1, !params) -> !ciphertext
    %e2 = jaxite.constant %1, %params : (i1, !params) -> !ciphertext
    %e3 = jaxite.constant %2, %params : (i1, !params) -> !ciphertext

    %l1 = jaxite.lut3_args %e1, %e2, %e3, %truth_table :(!ciphertext, !ciphertext, !ciphertext, i8) -> !jaxite.pmap_lut3_tuple
    %lut3_args_tensor = tensor.from_elements %l1 : tensor<1x!jaxite.pmap_lut3_tuple>

    %out = jaxite.pmap_lut3 %lut3_args_tensor, %bsks, %params : (tensor<1x!jaxite.pmap_lut3_tuple>, !bsks, !params) -> !ciphertext
    return
  }
}
