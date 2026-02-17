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
  // CHECK: func @test_new_params_from_literal
  func.func @test_new_params_from_literal() {
    // CHECK: %[[v1:.*]] = lattigo.bgv.new_parameters_from_literal
    %params = lattigo.bgv.new_parameters_from_literal {paramsLiteral = #paramsLiteral} : () -> !params
    // CHECK: %[[v2:.*]] = lattigo.bgv.new_parameters_from_literal
    %params2 = lattigo.bgv.new_parameters_from_literal {paramsLiteral = #paramsLiteral2} : () -> !params
    return
  }

  // CHECK: func @test_bgv_new_encoder
  func.func @test_bgv_new_encoder(%params: !params) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.new_encoder
    %encoder = lattigo.bgv.new_encoder %params : (!params) -> !encoder
    return
  }

  // CHECK: func @test_bgv_new_evaluator_no_key_set
  func.func @test_bgv_new_evaluator_no_key_set(%params: !params) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.new_evaluator
    %evaluator = lattigo.bgv.new_evaluator %params : (!params) -> !evaluator
    return
  }

  // CHECK: func @test_bgv_new_evaluator
  func.func @test_bgv_new_evaluator(%params: !params, %eval_key_set: !eval_key_set) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.new_evaluator
    %evaluator = lattigo.bgv.new_evaluator %params, %eval_key_set : (!params, !eval_key_set) -> !evaluator
    return
  }

  // CHECK: func @test_bgv_new_plaintext
  func.func @test_bgv_new_plaintext(%params: !params) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.new_plaintext
    %pt = lattigo.bgv.new_plaintext %params : (!params) -> !pt
    return
  }

  // CHECK: func @test_bgv_encode
  func.func @test_bgv_encode(%encoder: !encoder, %value : !value, %pt: !pt) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.encode
    %encoded = lattigo.bgv.encode %encoder, %value, %pt : (!encoder, !value, !pt) -> !pt
    return
  }

  // CHECK: func @test_bgv_add_new
  func.func @test_bgv_add_new(%evaluator: !evaluator, %lhs: !ct, %rhs: !ct) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.add_new
    %output = lattigo.bgv.add_new %evaluator, %lhs, %rhs : (!evaluator, !ct, !ct) -> !ct
    return
  }

  // CHECK: func @test_bgv_sub_new
  func.func @test_bgv_sub_new(%evaluator: !evaluator, %lhs: !ct, %rhs: !ct) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.sub_new
    %output = lattigo.bgv.sub_new %evaluator, %lhs, %rhs : (!evaluator, !ct, !ct) -> !ct
    return
  }

  // CHECK: func @test_bgv_mul_new
  func.func @test_bgv_mul_new(%evaluator: !evaluator, %lhs: !ct, %rhs: !ct) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.mul_new
    %output = lattigo.bgv.mul_new %evaluator, %lhs, %rhs : (!evaluator, !ct, !ct) -> !ct
    return
  }

  // CHECK: func @test_bgv_add
  func.func @test_bgv_add(%evaluator: !evaluator, %lhs: !ct, %rhs: !ct) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.add
    %output = lattigo.bgv.add %evaluator, %lhs, %rhs, %lhs : (!evaluator, !ct, !ct, !ct) -> !ct
    return
  }

  // CHECK: func @test_bgv_sub
  func.func @test_bgv_sub(%evaluator: !evaluator, %lhs: !ct, %rhs: !ct) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.sub
    %output = lattigo.bgv.sub %evaluator, %lhs, %rhs, %lhs : (!evaluator, !ct, !ct, !ct) -> !ct
    return
  }

  // CHECK: func @test_bgv_mul
  func.func @test_bgv_mul(%evaluator: !evaluator, %lhs: !ct, %rhs: !ct) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.mul
    %output = lattigo.bgv.mul %evaluator, %lhs, %rhs, %lhs : (!evaluator, !ct, !ct, !ct) -> !ct
    return
  }

  // CHECK: func @test_bgv_decode
  func.func @test_bgv_decode(%encoder: !encoder, %value : !value, %pt: !pt) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.decode
    %decoded = lattigo.bgv.decode %encoder, %pt, %value : (!encoder, !pt, !value) -> !value
    return
  }

  // CHECK: func @test_bgv_relinearize_new
  func.func @test_bgv_relinearize_new(%evaluator: !evaluator, %ct: !ct) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.relinearize_new
    %output = lattigo.bgv.relinearize_new %evaluator, %ct : (!evaluator, !ct) -> !ct
    return
  }

  // CHECK: func @test_bgv_rescale_new
  func.func @test_bgv_rescale_new(%evaluator: !evaluator, %ct: !ct) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.rescale_new
    %output = lattigo.bgv.rescale_new %evaluator, %ct : (!evaluator, !ct) -> !ct
    return
  }

  // CHECK: func @test_bgv_rotate_columns_new
  func.func @test_bgv_rotate_columns_new(%evaluator: !evaluator, %ct: !ct) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.rotate_columns_new
    %output = lattigo.bgv.rotate_columns_new %evaluator, %ct {static_shift = 1} : (!evaluator, !ct) -> !ct
    return
  }

  // CHECK: func @test_bgv_rotate_rows_new
  func.func @test_bgv_rotate_rows_new(%evaluator: !evaluator, %ct: !ct) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.rotate_rows_new
    %output = lattigo.bgv.rotate_rows_new %evaluator, %ct : (!evaluator, !ct) -> !ct
    return
  }

  // CHECK: func @test_bgv_relinearize
  func.func @test_bgv_relinearize(%evaluator: !evaluator, %ct: !ct) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.relinearize
    %output = lattigo.bgv.relinearize %evaluator, %ct, %ct : (!evaluator, !ct, !ct) -> !ct
    return
  }

  // CHECK: func @test_bgv_rescale
  func.func @test_bgv_rescale(%evaluator: !evaluator, %ct: !ct) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.rescale
    %output = lattigo.bgv.rescale %evaluator, %ct, %ct : (!evaluator, !ct, !ct) -> !ct
    return
  }

  // CHECK: func @test_bgv_rotate_columns
  func.func @test_bgv_rotate_columns(%evaluator: !evaluator, %ct: !ct) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.rotate_columns
    %output = lattigo.bgv.rotate_columns %evaluator, %ct, %ct {static_shift = 1} : (!evaluator, !ct, !ct) -> !ct
    return
  }

  // CHECK: func @test_bgv_rotate_rows
  func.func @test_bgv_rotate_rows(%evaluator: !evaluator, %ct: !ct) {
    // CHECK: %[[v1:.*]] = lattigo.bgv.rotate_rows
    %output = lattigo.bgv.rotate_rows %evaluator, %ct, %ct : (!evaluator, !ct, !ct) -> !ct
    return
  }
}
