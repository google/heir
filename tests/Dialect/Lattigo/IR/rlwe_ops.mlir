// RUN: heir-opt %s | FileCheck %s

// This simply tests for syntax.

!pk = !lattigo.rlwe.public_key
!sk = !lattigo.rlwe.secret_key
!rk = !lattigo.rlwe.relinearization_key
!gk5 = !lattigo.rlwe.galois_key<galoisElement = 5>
!eval_key_set = !lattigo.rlwe.evaluation_key_set

!ct = !lattigo.rlwe.ciphertext
!pt = !lattigo.rlwe.plaintext

!encryptor = !lattigo.rlwe.encryptor<publicKey = true>
!encryptor_sk = !lattigo.rlwe.encryptor<publicKey = false>
!decryptor = !lattigo.rlwe.decryptor
!key_generator = !lattigo.rlwe.key_generator

!params = !lattigo.bgv.parameter

module {
  // CHECK-LABEL: func @test_rlwe_new_key_generator
  func.func @test_rlwe_new_key_generator(%params: !params) {
    // CHECK: %[[v1:.*]] = lattigo.rlwe.new_key_generator
    %key_generator = lattigo.rlwe.new_key_generator %params : (!params) -> !key_generator
    return
  }

  // CHECK-LABEL: func @test_rlwe_gen_key_pair
  func.func @test_rlwe_gen_key_pair(%key_generator: !key_generator) {
    // CHECK: %[[v1:.*]], %[[v2:.*]] = lattigo.rlwe.gen_key_pair
    %sk, %pk = lattigo.rlwe.gen_key_pair %key_generator : (!key_generator) -> (!sk, !pk)
    return
  }

  // CHECK-LABEL: func @test_rlwe_gen_relinearization_key
  func.func @test_rlwe_gen_relinearization_key(%key_generator: !key_generator, %sk: !sk) {
    // CHECK: %[[v1:.*]] = lattigo.rlwe.gen_relinearization_key
    %rk = lattigo.rlwe.gen_relinearization_key %key_generator, %sk : (!key_generator, !sk) -> !rk
    return
  }

  // CHECK-LABEL: func @test_rlwe_gen_galois_key
  func.func @test_rlwe_gen_galois_key(%key_generator: !key_generator, %sk: !sk) {
    // CHECK: %[[v1:.*]] = lattigo.rlwe.gen_galois_key
    %gk = lattigo.rlwe.gen_galois_key %key_generator, %sk {galoisElement = 5} : (!key_generator, !sk) -> !gk5
    return
  }

  // CHECK-LABEL: func @test_rlwe_new_evaluation_key_set
  func.func @test_rlwe_new_evaluation_key_set(%rk : !rk, %gk5 : !gk5) {
    // CHECK: %[[v1:.*]] = lattigo.rlwe.new_evaluation_key_set
    %eval_key_set = lattigo.rlwe.new_evaluation_key_set %rk, %gk5 : (!rk, !gk5) -> !eval_key_set
    return
  }

  // CHECK-LABEL: func @test_rlwe_new_encryptor_pk
  func.func @test_rlwe_new_encryptor_pk(%params: !params, %pk: !pk) {
    // CHECK: %[[v1:.*]] = lattigo.rlwe.new_encryptor
    %encryptor = lattigo.rlwe.new_encryptor %params, %pk : (!params, !pk) -> !encryptor
    return
  }

  // CHECK-LABEL: func @test_rlwe_new_encryptor_sk
  func.func @test_rlwe_new_encryptor_sk(%params: !params, %sk: !sk) {
    // CHECK: %[[v1:.*]] = lattigo.rlwe.new_encryptor
    %encryptor_sk = lattigo.rlwe.new_encryptor %params, %sk : (!params, !sk) -> !encryptor_sk
    return
  }

  // CHECK-LABEL: func @test_rlwe_new_decryptor
  func.func @test_rlwe_new_decryptor(%params: !params, %sk: !sk) {
    // CHECK: %[[v1:.*]] = lattigo.rlwe.new_decryptor
    %decryptor = lattigo.rlwe.new_decryptor %params, %sk : (!params, !sk) -> !decryptor
    return
  }

  // CHECK-LABEL: func @test_rlwe_encrypt
  func.func @test_rlwe_encrypt(%encryptor: !encryptor, %pt: !pt) {
    // CHECK: %[[v1:.*]] = lattigo.rlwe.encrypt
    %ct = lattigo.rlwe.encrypt %encryptor, %pt : (!encryptor, !pt) -> !ct
    return
  }

  // CHECK-LABEL: func @test_rlwe_decrypt
  func.func @test_rlwe_decrypt(%decryptor: !decryptor, %ct: !ct) {
    // CHECK: %[[v1:.*]] = lattigo.rlwe.decrypt
    %pt = lattigo.rlwe.decrypt %decryptor, %ct : (!decryptor, !ct) -> !pt
    return
  }
}
