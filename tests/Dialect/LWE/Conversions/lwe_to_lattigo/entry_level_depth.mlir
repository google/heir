// RUN: heir-opt --lwe-to-lattigo %s | FileCheck %s

// A lattigo ciphertext is opaque, so the level an argument starts at has to be
// recorded here or it is lost. The level is measured against the module's Q
// chain: an LWE type's own element list can be a truncated view of the chain,
// and measuring against that understates the level.

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>

!rns_L0 = !rns.rns<!mod_arith.int<36028797018652673 : i64>>
!rns_L2 = !rns.rns<!mod_arith.int<36028797018652673 : i64>, !mod_arith.int<35184372121601 : i64>, !mod_arith.int<35184372744193 : i64>>
#ring_rns_L0 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**1024>>
#ring_rns_L2 = #polynomial.ring<coefficientType = !rns_L2, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0, encryption_type = mix>
#ciphertext_space_L2 = #lwe.ciphertext_space<ring = #ring_rns_L2, encryption_type = mix>

// The fully consumed value carries only one chain element, so its own list puts
// its "top" at level 0 and would report a depth of 0.
#chain_truncated = #lwe.modulus_chain<elements = <36028797018652673 : i64>, current = 0>
#chain_full = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64, 35184372744193 : i64>, current = 2>

!ct_L0 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #chain_truncated>
!ct_L2 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L2, key = #key, modulus_chain = #chain_full>

// CHECK-DAG: ![[ctType:.*]] = !lattigo.rlwe.ciphertext

module attributes {backend.lattigo, ckks.schemeParam = #ckks.scheme_param<logN = 10, Q = [36028797018652673, 35184372121601, 35184372744193], P = [1152921504606994433], logDefaultScale = 45>, scheme.ckks} {
  // Q has three moduli, so the top of the chain is level 2: the argument at
  // level 2 is unannotated and the one at level 0 records a depth of 2.
  // CHECK: func.func @entry_levels
  // CHECK-SAME: %{{[a-z_0-9]+}}: ![[ctType]], %{{[a-z_0-9]+}}: ![[ctType]] {lwe.entry_level_depth = 2 : i64})
  func.func @entry_levels(%fresh: !ct_L2, %deep: !ct_L0) -> !ct_L2 {
    return %fresh : !ct_L2
  }
}
