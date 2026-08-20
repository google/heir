// This file reproduces an absolute-level mismatch in Lattigo destination
// buffer reuse after a prepared linear transform.
// RUN: heir-opt --scheme-to-lattigo %s | FileCheck %s

!Z36028797018652673_i64 = !mod_arith.int<36028797018652673 : i64>
!Z35184372121601_i64 = !mod_arith.int<35184372121601 : i64>
!Z35184388898817_i64 = !mod_arith.int<35184388898817 : i64>

#encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
#modulus_chain_L2_C0 = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64, 35184388898817 : i64>, current = 0>
#modulus_chain_L2_C1 = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64, 35184388898817 : i64>, current = 1>
#modulus_chain_L2_C2 = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64, 35184388898817 : i64>, current = 2>
#ring_f64 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**8192>>
!rns_L0 = !rns.rns<!Z36028797018652673_i64>
!rns_L1 = !rns.rns<!Z36028797018652673_i64, !Z35184372121601_i64>
!rns_L2 = !rns.rns<!Z36028797018652673_i64, !Z35184372121601_i64, !Z35184388898817_i64>
#ring_rns_L0 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**8192>>
#ring_rns_L1 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**8192>>
#ring_rns_L2 = #polynomial.ring<coefficientType = !rns_L2, polynomialModulus = <1 + x**8192>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0, encryption_type = mix>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1, encryption_type = mix>
#ciphertext_space_L2 = #lwe.ciphertext_space<ring = #ring_rns_L2, encryption_type = mix>
!ct_L0 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64, encoding = #encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain_L2_C0>
!ct_L1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64, encoding = #encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L2_C1>
!ct_L2 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64, encoding = #encoding>, ciphertext_space = #ciphertext_space_L2, key = #key, modulus_chain = #modulus_chain_L2_C2>
!prepared_L2 = !kernel.prepared_linear_transform<level = 2, slots = 1024, log_bsgs_ratio = 0>

module attributes {
  backend.lattigo,
  ckks.schemeParam = #ckks.scheme_param<
    logN = 12,
    Q = [36028797018652673, 35184372121601, 35184388898817],
    P = [1152921504606994433],
    logDefaultScale = 45>,
  scheme.actual_slot_count = 2048 : i64,
  scheme.ckks,
  scheme.requested_slot_count = 1024 : i64
} {
  // The dead low path leaves an L0 allocation at consumed depth 2. It must not
  // become the destination of the L1 transform result. The dead bootstrap
  // output remains at L2 and is safe to reuse: the emitter first gives the
  // receiver the operand's runtime level, then Rescale shrinks it to L1.
  // CHECK: func.func @different_absolute_levels
  // CHECK: %[[LOW:.*]] = lattigo.rlwe.drop_level_new
  // CHECK: %[[BOOT:.*]] = lattigo.ckks.bootstrap
  // CHECK: %[[APPLY:.*]] = lattigo.ckks.apply_linear_transform {{.*}}%[[BOOT]]
  // CHECK: %[[HIGH:.*]] = lattigo.ckks.rescale {{.*}}%[[APPLY]], %[[BOOT]]
  // CHECK-NOT: lattigo.ckks.rescale {{.*}}%[[APPLY]], %[[LOW]]
  // CHECK: lattigo.ckks.add {{.*}}%[[HIGH]], %[[HIGH]], %[[BOOT]]
  func.func @different_absolute_levels(
      %base: !ct_L2, %boot_input: !ct_L0,
      %prepared: !prepared_L2) -> !ct_L1 {
    %low = ckks.level_reduce %base {levelToDrop = 2 : i64} : !ct_L2 -> !ct_L0
    %low_used = ckks.add %low, %boot_input : (!ct_L0, !ct_L0) -> !ct_L0
    %low_sink = ckks.add %low_used, %boot_input : (!ct_L0, !ct_L0) -> !ct_L0
    %fresh = ckks.bootstrap %boot_input : !ct_L0 -> !ct_L2
    %high = kernel.apply_linear_transform %fresh, %prepared : !ct_L2, !prepared_L2 -> !ct_L1
    %out = ckks.add %high, %high : (!ct_L1, !ct_L1) -> !ct_L1
    return %out : !ct_L1
  }
}
