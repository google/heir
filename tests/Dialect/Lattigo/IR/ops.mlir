// RUN: heir-opt %s | FileCheck %s

// This simply tests for syntax.

!pk = !lattigo.rlwe.public_key
!sk = !lattigo.rlwe.secret_key
!ct = !lattigo.rlwe.ciphertext
!pt = !lattigo.rlwe.plaintext

!encryptor = !lattigo.rlwe.encryptor
!decryptor = !lattigo.rlwe.decryptor
!key_generator = !lattigo.rlwe.key_generator

!evaluator = !lattigo.bgv.evaluator
!encoder = !lattigo.bgv.encoder
!params = !lattigo.bgv.parameter

!value = tensor<8xi32>

#paramsLiteral = #lattigo.bgv.parameters_literal<
    logN = 14,
    logQ = [56, 55, 55],
    logP = [55],
    plaintextModulus = 0x3ee0001
>

#paramsLiteral2 = #lattigo.bgv.parameters_literal<
    logN = 14,
    Q = [65537, 17, 23],
    P = [29],
    plaintextModulus = 0x3ee0001
>

module {
  // CHECK-LABEL: func @test_new_params_from_literal
  func.func @test_new_params_from_literal() {
    // CHECK: %[[v1:.*]] = lattigo.bgv.new_parameters_from_literal
    %params = lattigo.bgv.new_parameters_from_literal {paramsLiteral = #paramsLiteral} : () -> !params
    // CHECK: %[[v2:.*]] = lattigo.bgv.new_parameters_from_literal
    %params2 = lattigo.bgv.new_parameters_from_literal {paramsLiteral = #paramsLiteral2} : () -> !params
    return
  }

  // CHECK-LABEL: func @test_bgv_new_encoder
  func.func @test_bgv_new_encoder(%params: !params) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.new_encoder
    %encoder = lattigo.bgv.new_encoder %params : (!params) -> !encoder
    return
  }

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

  // CHECK-LABEL: func @test_rlwe_new_encryptor
  func.func @test_rlwe_new_encryptor(%params: !params, %pk: !pk) {
    // CHECK: %[[v1:.*]] = lattigo.rlwe.new_encryptor
    %encryptor = lattigo.rlwe.new_encryptor %params, %pk : (!params, !pk) -> !encryptor
    return
  }

  // CHECK-LABEL: func @test_rlwe_new_decryptor
  func.func @test_rlwe_new_decryptor(%params: !params, %sk: !sk) {
    // CHECK: %[[v1:.*]] = lattigo.rlwe.new_decryptor
    %decryptor = lattigo.rlwe.new_decryptor %params, %sk : (!params, !sk) -> !decryptor
    return
  }

  // CHECK-LABEL: func @test_bgv_new_evaluator
  func.func @test_bgv_new_evaluator(%params: !params) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.new_evaluator
    %evaluator = lattigo.bgv.new_evaluator %params : (!params) -> !evaluator
    return
  }

  // CHECK-LABEL: func @test_bgv_new_plaintext
  func.func @test_bgv_new_plaintext(%params: !params) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.new_plaintext
    %pt = lattigo.bgv.new_plaintext %params : (!params) -> !pt
    return
  }

  // CHECK-LABEL: func @test_bgv_encode
  func.func @test_bgv_encode(%encoder: !encoder, %value : !value, %pt: !pt) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.encode
    %encoded = lattigo.bgv.encode %encoder, %value, %pt : (!encoder, !value, !pt) -> !pt
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

  // CHECK-LABEL: func @test_bgv_add
  func.func @test_bgv_add(%evaluator: !evaluator, %lhs: !ct, %rhs: !ct) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.add
    %output = lattigo.bgv.add %evaluator, %lhs, %rhs : (!evaluator, !ct, !ct) -> !ct
    return
  }

  // CHECK-LABEL: func @test_bgv_sub
  func.func @test_bgv_sub(%evaluator: !evaluator, %lhs: !ct, %rhs: !ct) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.sub
    %output = lattigo.bgv.sub %evaluator, %lhs, %rhs : (!evaluator, !ct, !ct) -> !ct
    return
  }

  // CHECK-LABEL: func @test_bgv_mul
  func.func @test_bgv_mul(%evaluator: !evaluator, %lhs: !ct, %rhs: !ct) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.mul
    %output = lattigo.bgv.mul %evaluator, %lhs, %rhs : (!evaluator, !ct, !ct) -> !ct
    return
  }

  // CHECK-LABEL: func @test_bgv_decode
  func.func @test_bgv_decode(%encoder: !encoder, %value : !value, %pt: !pt) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.decode
    %decoded = lattigo.bgv.decode %encoder, %pt, %value : (!encoder, !pt, !value) -> !value
    return
  }
}
