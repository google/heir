// RUN: heir-opt %s | FileCheck %s

// This simply tests for syntax.

!bsks = !jaxite.server_key_set
!params = !jaxite.params
#unspecified_encoding = #lwe.unspecified_bit_field_encoding<
  cleartext_bitwidth=3>
#params = #lwe.lwe_params<cmod = 7917, dimension = 10>
!ciphertext = !lwe.lwe_ciphertext<encoding = #unspecified_encoding, lwe_params = #params>

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
