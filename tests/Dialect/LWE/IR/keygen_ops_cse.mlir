// RUN: heir-opt --cse --canonicalize %s | FileCheck %s

// Key generation samples fresh randomness, so identical key generation ops
// must not be merged, but an unused key can be removed.

!Zq0 = !mod_arith.int<1095233372161 : i64>
!rns_Q = !rns.rns<!Zq0>
#ring_Q = #polynomial.ring<coefficientType = !rns_Q, polynomialModulus = <1 + x**1024>>
!sk = !lwe.lwe_secret_key<key = #lwe.key<>, ring = #ring_Q>
!pk = !lwe.lwe_public_key<key = #lwe.key<>, ring = #ring_Q>

// CHECK: @test_not_merged
func.func @test_not_merged() -> (!sk, !sk, !pk, !pk) {
  // CHECK-COUNT-2: lwe.gen_secret_key
  // CHECK-COUNT-2: lwe.gen_public_key
  %sk0 = lwe.gen_secret_key : () -> !sk
  %sk1 = lwe.gen_secret_key : () -> !sk
  %pk0 = lwe.gen_public_key %sk0 : (!sk) -> !pk
  %pk1 = lwe.gen_public_key %sk0 : (!sk) -> !pk
  return %sk0, %sk1, %pk0, %pk1 : !sk, !sk, !pk, !pk
}

// CHECK: @test_unused_removed
func.func @test_unused_removed() {
  // CHECK-NOT: lwe.gen_
  %sk = lwe.gen_secret_key : () -> !sk
  %pk = lwe.gen_public_key %sk : (!sk) -> !pk
  // CHECK: return
  return
}
