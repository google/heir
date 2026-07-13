// RUN: heir-opt --cse %s | FileCheck %s

!pk = !lattigo.rlwe.public_key
!eval_key_set = !lattigo.rlwe.evaluation_key_set
!ct = !lattigo.rlwe.ciphertext
!pt = !lattigo.rlwe.plaintext

!bgv_evaluator = !lattigo.bgv.evaluator
!bgv_params = !lattigo.bgv.parameter

!ckks_evaluator = !lattigo.ckks.bootstrapping_evaluator

#paramsLiteral = #lattigo.bgv.parameters_literal<
    logN = 14,
    logQ = [56, 55, 55],
    logP = [55],
    plaintextModulus = 0x3ee0001
>

module {
  // CHECK-LABEL: @test_bgv_new_plaintext_cse
  func.func @test_bgv_new_plaintext_cse() {
    %params = lattigo.bgv.new_parameters_from_literal {paramsLiteral = #paramsLiteral} : () -> !bgv_params
    // Allocations should not be deduplicated.
    // CHECK: lattigo.bgv.new_plaintext
    // CHECK: lattigo.bgv.new_plaintext
    %pt1 = lattigo.bgv.new_plaintext %params : (!bgv_params) -> !pt
    %pt2 = lattigo.bgv.new_plaintext %params : (!bgv_params) -> !pt
    return
  }

  // CHECK-LABEL: @test_bgv_add_cse
  func.func @test_bgv_add_cse(%evaluator: !bgv_evaluator, %lhs: !ct, %rhs: !ct) {
    // In-place operations should not be deduplicated because they mutate their inplace operand.
    // CHECK: lattigo.bgv.add
    // CHECK: lattigo.bgv.add
    %output1 = lattigo.bgv.add %evaluator, %lhs, %rhs, %lhs : (!bgv_evaluator, !ct, !ct, !ct) -> !ct
    %output2 = lattigo.bgv.add %evaluator, %lhs, %rhs, %lhs : (!bgv_evaluator, !ct, !ct, !ct) -> !ct
    return
  }

  // CHECK-LABEL: @test_ckks_bootstrap_cse
  func.func @test_ckks_bootstrap_cse(%evaluator: !ckks_evaluator, %ct: !ct) {
    // In-place operations should not be deduplicated.
    // CHECK: lattigo.ckks.bootstrap
    // CHECK: lattigo.ckks.bootstrap
    %output1 = lattigo.ckks.bootstrap %evaluator, %ct, %ct : (!ckks_evaluator, !ct, !ct) -> !ct
    %output2 = lattigo.ckks.bootstrap %evaluator, %ct, %ct : (!ckks_evaluator, !ct, !ct) -> !ct
    return
  }
}
