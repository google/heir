// RUN: heir-opt %s | FileCheck %s

// This simply tests for syntax.

!pk = !lattigo.rlwe.public_key
!sk = !lattigo.rlwe.secret_key
!rk = !lattigo.rlwe.relinearization_key
!gk5 = !lattigo.rlwe.galois_key<galoisElement = 5>
!eval_key_set = !lattigo.rlwe.evaluation_key_set

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

  // CHECK-LABEL: func @test_bgv_new_evaluator_no_key_set
  func.func @test_bgv_new_evaluator_no_key_set(%params: !params) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.new_evaluator
    %evaluator = lattigo.bgv.new_evaluator %params : (!params) -> !evaluator
    return
  }

  // CHECK-LABEL: func @test_bgv_new_evaluator
  func.func @test_bgv_new_evaluator(%params: !params, %eval_key_set: !eval_key_set) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.new_evaluator
    %evaluator = lattigo.bgv.new_evaluator %params, %eval_key_set : (!params, !eval_key_set) -> !evaluator
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

  // CHECK-LABEL: func @test_bgv_relinearize
  func.func @test_bgv_relinearize(%evaluator: !evaluator, %ct: !ct) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.relinearize
    %output = lattigo.bgv.relinearize %evaluator, %ct : (!evaluator, !ct) -> !ct
    return
  }

  // CHECK-LABEL: func @test_bgv_rescale
  func.func @test_bgv_rescale(%evaluator: !evaluator, %ct: !ct) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.rescale
    %output = lattigo.bgv.rescale %evaluator, %ct : (!evaluator, !ct) -> !ct
    return
  }

  // CHECK-LABEL: func @test_bgv_rotate_columns
  func.func @test_bgv_rotate_columns(%evaluator: !evaluator, %ct: !ct) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.rotate_columns
    %output = lattigo.bgv.rotate_columns %evaluator, %ct {offset = 1} : (!evaluator, !ct) -> !ct
    return
  }

  // CHECK-LABEL: func @test_bgv_rotate_rows
  func.func @test_bgv_rotate_rows(%evaluator: !evaluator, %ct: !ct) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.rotate_rows
    %output = lattigo.bgv.rotate_rows %evaluator, %ct : (!evaluator, !ct) -> !ct
    return
  }
}
