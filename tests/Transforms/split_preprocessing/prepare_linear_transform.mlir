// RUN: heir-opt --split-preprocessing %s | FileCheck %s

// A kernel.prepare_linear_transform is an encode-like op
// (PlaintextEncodeOpInterface), so split-preprocessing hoists it (and the
// cleartext diagonals feeding it) into the __preprocessing helper; the
// evaluation function only applies the loaded prepared transform.

!prepared = !kernel.prepared_linear_transform<level = 0, slots = 512, log_bsgs_ratio = 0>

// CHECK: func.func @prepare__preprocessing() -> !preprocessing.storage<!kernel.prepared_linear_transform<level = 0, slots = 512, log_bsgs_ratio = 0>>
// CHECK-SAME: server.preprocessing_func = {entry_arg_indices = array<i64>, func_name = "prepare"}
// CHECK: %[[LT:.*]] = kernel.prepare_linear_transform
// CHECK: preprocessing.store %[[LT]]

// CHECK: func.func @prepare__preprocessed(%[[ct:.*]]: ![[ct_ty:.*]], %[[storage:.*]]: !preprocessing.storage
// CHECK-NOT: kernel.prepare_linear_transform
// CHECK: %[[LOAD:.*]] = preprocessing.load %[[storage]][] site 0
// CHECK: kernel.apply_linear_transform %[[ct]], %[[LOAD]]

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
#modulus_chain = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 0>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L0 = !rns.rns<!mod_arith.int<36028797018652673 : i64>>
#ring_rns_L0_1_x1024 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024, encryption_type = mix>
!ct = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain>

module attributes {backend.lattigo, scheme.ckks} {
  func.func @prepare(%ct: !ct) -> !ct {
    %diagonals = arith.constant dense<1.0> : tensor<2x512xf64>
    %lt = kernel.prepare_linear_transform %diagonals {
      diagonal_indices = array<i64: 0, 1>
    } : tensor<2x512xf64> -> !prepared
    %0 = kernel.apply_linear_transform %ct, %lt : !ct, !prepared -> !ct
    return %0 : !ct
  }
}
