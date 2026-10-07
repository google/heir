// RUN: heir-opt --lwe-to-lattigo --split-input-file %s | FileCheck %s

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
#modulus_chain_L1 = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 1>
#modulus_chain_L0 = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 0>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L1 = !rns.rns<!mod_arith.int<36028797018652673 : i64>, !mod_arith.int<35184372121601 : i64>>
#ring_rns_L1_1_x1024 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024, encryption_type = mix>
!ct_L1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L1>
!rns_L0 = !rns.rns<!mod_arith.int<36028797018652673 : i64>>
#ring_rns_L0_1_x1024 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024, encryption_type = mix>
!ct_L0 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain_L0>
!prepared = !kernel.prepared_linear_transform<level = 1, slots = 512, log_bsgs_ratio = 0>

// CHECK-DAG: ![[CT:.*]] = !lattigo.rlwe.ciphertext
// CHECK-DAG: ![[ENC:.*]] = !lattigo.ckks.encoder
// CHECK-DAG: ![[EVAL:.*]] = !lattigo.ckks.evaluator
// CHECK-DAG: ![[PARAM:.*]] = !lattigo.ckks.parameter
// CHECK-DAG: ![[LT:.*]] = !lattigo.ckks.linear_transformation

module attributes {backend.lattigo, scheme.ckks} {
  // CHECK: func.func @test_prepare_apply(%[[EVAL:.*]]: ![[EVAL]], %[[PARAM:.*]]: ![[PARAM]], %[[ENC:.*]]: ![[ENC]], %[[CT:.*]]: ![[CT]]{{.*}}) -> ![[CT]]
  // CHECK: %[[DIAGS:.*]] = arith.constant dense<1.000000e+00> : tensor<2x512xf64>
  // CHECK: %[[PREPARED:.*]] = lattigo.ckks.prepare_linear_transform %[[PARAM]], %[[ENC]], %[[DIAGS]] <diagonal_indices = [0, 1], levelQ = 1 : i64, logSlots = 9 : i64, logBabyStepGiantStepRatio = 0 : i64> : (![[PARAM]], ![[ENC]], tensor<2x512xf64>) -> ![[LT]]
  // CHECK: %[[APPLY:.*]] = lattigo.ckks.apply_linear_transform %[[EVAL]], %[[CT]], %[[PREPARED]] : (![[EVAL]], ![[CT]], ![[LT]]) -> ![[CT]]
  // CHECK-NEXT: %[[RESCALE:.*]] = lattigo.ckks.rescale_new %[[EVAL]], %[[APPLY]] : (![[EVAL]], ![[CT]]) -> ![[CT]]
  // CHECK-NOT: lattigo.ckks.rescale_new
  // CHECK: return %[[RESCALE]] : ![[CT]]
  func.func @test_prepare_apply(%ct: !ct_L1) -> !ct_L0 {
    %diagonals = arith.constant dense<1.000000e+00> : tensor<2x512xf64>
    %lt = kernel.prepare_linear_transform %diagonals <diagonal_indices = [0, 1]> : tensor<2x512xf64> -> !prepared
    %0 = kernel.apply_linear_transform %ct, %lt : !ct_L1, !prepared -> !ct_L0
    return %0 : !ct_L0
  }
}

// -----

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
#modulus_chain_L1 = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 1>
#modulus_chain_L0 = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 0>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L1 = !rns.rns<!mod_arith.int<36028797018652673 : i64>, !mod_arith.int<35184372121601 : i64>>
#ring_rns_L1_1_x1024 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024, encryption_type = mix>
!ct_L1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L1>
!rns_L0 = !rns.rns<!mod_arith.int<36028797018652673 : i64>>
#ring_rns_L0_1_x1024 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024, encryption_type = mix>
!ct_L0 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain_L0>
!prepared = !kernel.prepared_linear_transform<level = 1, slots = 512, log_bsgs_ratio = 0>

// CHECK-DAG: ![[CT:.*]] = !lattigo.rlwe.ciphertext
// CHECK-DAG: ![[ENC:.*]] = !lattigo.ckks.encoder
// CHECK-DAG: ![[EVAL:.*]] = !lattigo.ckks.evaluator
// CHECK-DAG: ![[PARAM:.*]] = !lattigo.ckks.parameter
// CHECK-DAG: ![[LT:.*]] = !lattigo.ckks.linear_transformation

module attributes {backend.lattigo, scheme.ckks} {
  // CHECK: func.func @test_preprocessing_helpers__preprocessing(%[[PARAM:.*]]: ![[PARAM]], %[[ENC:.*]]: ![[ENC]]) -> !preprocessing.storage<![[LT]]>
  // CHECK: %[[DIAGS:.*]] = arith.constant dense<1.000000e+00> : tensor<2x512xf64>
  // CHECK: %[[PREPARED:.*]] = lattigo.ckks.prepare_linear_transform %[[PARAM]], %[[ENC]], %[[DIAGS]] <diagonal_indices = [0, 1], levelQ = 1 : i64, logSlots = 9 : i64, logBabyStepGiantStepRatio = 0 : i64> : (![[PARAM]], ![[ENC]], tensor<2x512xf64>) -> ![[LT]]
  // CHECK: %[[STORAGE:.*]] = preprocessing.empty : <![[LT]]>
  // CHECK: preprocessing.store %[[PREPARED]], %[[STORAGE]][] site 0<![[LT]]> : ![[LT]], <![[LT]]>
  // CHECK: return %[[STORAGE]]
  func.func @test_preprocessing_helpers__preprocessing() -> !preprocessing.storage<!prepared> {
    %diagonals = arith.constant dense<1.000000e+00> : tensor<2x512xf64>
    %lt = kernel.prepare_linear_transform %diagonals <diagonal_indices = [0, 1]> : tensor<2x512xf64> -> !prepared
    %storage = preprocessing.empty : !preprocessing.storage<!prepared>
    preprocessing.store %lt, %storage[] site 0 <!prepared> : !prepared, !preprocessing.storage<!prepared>
    return %storage : !preprocessing.storage<!prepared>
  }

  // CHECK: func.func @test_preprocessing_helpers__preprocessed(%[[EVAL:.*]]: ![[EVAL]], %[[PARAM:.*]]: ![[PARAM]], %[[ENC:.*]]: ![[ENC]], %[[CT:.*]]: ![[CT]]{{.*}}, %[[STORAGE:.*]]: !preprocessing.storage<![[LT]]>) -> ![[CT]]
  // CHECK: %[[LOAD:.*]] = preprocessing.load %[[STORAGE]][] site 0<![[LT]]> : <![[LT]]>, ![[LT]]
  // CHECK: %[[APPLY:.*]] = lattigo.ckks.apply_linear_transform %[[EVAL]], %[[CT]], %[[LOAD]] : (![[EVAL]], ![[CT]], ![[LT]]) -> ![[CT]]
  // CHECK-NEXT: %[[RESCALE:.*]] = lattigo.ckks.rescale_new %[[EVAL]], %[[APPLY]] : (![[EVAL]], ![[CT]]) -> ![[CT]]
  // CHECK-NOT: lattigo.ckks.rescale_new
  // CHECK: return %[[RESCALE]] : ![[CT]]
  func.func @test_preprocessing_helpers__preprocessed(%ct: !ct_L1, %storage: !preprocessing.storage<!prepared>) -> !ct_L0 {
    %lt = preprocessing.load %storage[] site 0 <!prepared> : !preprocessing.storage<!prepared>, !prepared
    %0 = kernel.apply_linear_transform %ct, %lt : !ct_L1, !prepared -> !ct_L0
    return %0 : !ct_L0
  }
}
