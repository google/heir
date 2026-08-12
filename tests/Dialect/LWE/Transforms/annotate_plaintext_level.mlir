// RUN: heir-opt --lwe-annotate-plaintext-level --split-input-file %s | FileCheck %s

!Z1005037682689_i64_ = !mod_arith.int<1005037682689 : i64>
!Z1032955396097_i64_ = !mod_arith.int<1032955396097 : i64>
!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z998595133441_i64_ = !mod_arith.int<998595133441 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain_L5_C1_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 1>
#modulus_chain_L5_C3_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 3>
!rns_L1_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_>
!rns_L3_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_, !Z1005037682689_i64_, !Z998595133441_i64_>
#ring_Z65537_i64_1_x32_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**32>>
#ring_rns_L1_1_x32_ = #polynomial.ring<coefficientType = !rns_L1_, polynomialModulus = <1 + x**32>>
#ring_rns_L3_1_x32_ = #polynomial.ring<coefficientType = !rns_L3_, polynomialModulus = <1 + x**32>>
#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x32_, encoding = #full_crt_packing_encoding>
#ciphertext_space_L1_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x32_, encryption_type = lsb>
#ciphertext_space_L3_ = #lwe.ciphertext_space<ring = #ring_rns_L3_1_x32_, encryption_type = lsb>
!pt = !lwe.lwe_plaintext<plaintext_space = #plaintext_space>
!pkey_L1_ = !lwe.lwe_public_key<key = #key, ring = #ring_rns_L1_1_x32_>
!pkey_L3_ = !lwe.lwe_public_key<key = #key, ring = #ring_rns_L3_1_x32_>
!ct_L1_ = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L5_C1_>
!ct_L3_ = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L3_, key = #key, modulus_chain = #modulus_chain_L5_C3_>

// CHECK: func @single_use
func.func @single_use(%ct: !ct_L1_, %value: tensor<32xi64>) -> !ct_L1_ {
  // CHECK: lwe.rlwe_encode
  // CHECK-SAME: level = 1 : i64
  %pt = lwe.rlwe_encode %value {encoding = #full_crt_packing_encoding, ring = #ring_Z65537_i64_1_x32_} : tensor<32xi64> -> !pt
  %res = lwe.rmul_plain %ct, %pt : (!ct_L1_, !pt) -> !ct_L1_
  return %res : !ct_L1_
}

// A plaintext shared by uses at several levels takes the highest of them, since
// combining it with the lower-level ciphertext drops to that ciphertext's level
// anyway.
// CHECK: func @multiple_uses
func.func @multiple_uses(%ct1: !ct_L1_, %ct3: !ct_L3_, %value: tensor<32xi64>) -> (!ct_L1_, !ct_L3_) {
  // CHECK: lwe.rlwe_encode
  // CHECK-SAME: level = 3 : i64
  %pt = lwe.rlwe_encode %value {encoding = #full_crt_packing_encoding, ring = #ring_Z65537_i64_1_x32_} : tensor<32xi64> -> !pt
  %res1 = lwe.rmul_plain %ct1, %pt : (!ct_L1_, !pt) -> !ct_L1_
  %res3 = lwe.rmul_plain %ct3, %pt : (!ct_L3_, !pt) -> !ct_L3_
  return %res1, %res3 : !ct_L1_, !ct_L3_
}

// The plaintext of an encryption is read off the resulting ciphertext. Unlike a
// ct-pt use this level is exact, not a lower bound: `rlwe_encrypt` has no
// ciphertext operand to be clamped against, so the ciphertext is built at
// whatever level the plaintext carries.
// CHECK: func @encrypt
func.func @encrypt(%value: tensor<32xi64>, %pk: !pkey_L1_) -> !ct_L1_ {
  // CHECK: lwe.rlwe_encode
  // CHECK-SAME: level = 1 : i64
  %pt = lwe.rlwe_encode %value {encoding = #full_crt_packing_encoding, ring = #ring_Z65537_i64_1_x32_} : tensor<32xi64> -> !pt
  %ct = lwe.rlwe_encrypt %pt, %pk : (!pt, !pkey_L1_) -> !ct_L1_
  return %ct : !ct_L1_
}

// An encryption's exact level also serves a ct-pt use at or below it, which
// only drops to the ciphertext's own level.
// CHECK: func @encrypt_covers_combine
func.func @encrypt_covers_combine(%ct1: !ct_L1_, %value: tensor<32xi64>, %pk: !pkey_L3_) -> (!ct_L3_, !ct_L1_) {
  // CHECK: lwe.rlwe_encode
  // CHECK-SAME: level = 3 : i64
  %pt = lwe.rlwe_encode %value {encoding = #full_crt_packing_encoding, ring = #ring_Z65537_i64_1_x32_} : tensor<32xi64> -> !pt
  %ct = lwe.rlwe_encrypt %pt, %pk : (!pt, !pkey_L3_) -> !ct_L3_
  %res = lwe.rmul_plain %ct1, %pt : (!ct_L1_, !pt) -> !ct_L1_
  return %ct, %res : !ct_L3_, !ct_L1_
}

// When a ct-pt use needs a level above the encryption's, no single level is
// correct: raising it to 3 would build the encrypted ciphertext one limb above
// the level its type declares, which is what the scale bookkeeping downstream
// is computed against. Leave it to the fallback.
// CHECK: func @encrypt_conflicts_with_combine
func.func @encrypt_conflicts_with_combine(%ct3: !ct_L3_, %value: tensor<32xi64>, %pk: !pkey_L1_) -> (!ct_L1_, !ct_L3_) {
  // CHECK: lwe.rlwe_encode
  // CHECK-NOT: level
  %pt = lwe.rlwe_encode %value {encoding = #full_crt_packing_encoding, ring = #ring_Z65537_i64_1_x32_} : tensor<32xi64> -> !pt
  %ct = lwe.rlwe_encrypt %pt, %pk : (!pt, !pkey_L1_) -> !ct_L1_
  %res = lwe.rmul_plain %ct3, %pt : (!ct_L3_, !pt) -> !ct_L3_
  return %ct, %res : !ct_L1_, !ct_L3_
}

// Two encryptions wanting different levels cannot both be served by one
// encoding either.
// CHECK: func @conflicting_encryptions
func.func @conflicting_encryptions(%value: tensor<32xi64>, %pk1: !pkey_L1_, %pk3: !pkey_L3_) -> (!ct_L1_, !ct_L3_) {
  // CHECK: lwe.rlwe_encode
  // CHECK-NOT: level
  %pt = lwe.rlwe_encode %value {encoding = #full_crt_packing_encoding, ring = #ring_Z65537_i64_1_x32_} : tensor<32xi64> -> !pt
  %ct1 = lwe.rlwe_encrypt %pt, %pk1 : (!pt, !pkey_L1_) -> !ct_L1_
  %ct3 = lwe.rlwe_encrypt %pt, %pk3 : (!pt, !pkey_L3_) -> !ct_L3_
  return %ct1, %ct3 : !ct_L1_, !ct_L3_
}

// Plaintexts are not always consumed directly, so the walk continues through
// intervening plaintext-typed values.
// CHECK: func @through_tensor
func.func @through_tensor(%ct: tensor<1x!ct_L1_>, %value: tensor<32xi64>) -> tensor<1x!ct_L1_> {
  // CHECK: lwe.rlwe_encode
  // CHECK-SAME: level = 1 : i64
  %pt = lwe.rlwe_encode %value {encoding = #full_crt_packing_encoding, ring = #ring_Z65537_i64_1_x32_} : tensor<32xi64> -> !pt
  %pts = tensor.from_elements %pt : tensor<1x!pt>
  %res = lwe.rmul_plain %ct, %pts : (tensor<1x!ct_L1_>, tensor<1x!pt>) -> tensor<1x!ct_L1_>
  return %res : tensor<1x!ct_L1_>
}

// With no ciphertext to read a level from, the encode op is left alone and the
// backend falls back to the top of the modulus chain.
// CHECK: func @no_ciphertext_use
func.func @no_ciphertext_use(%value: tensor<32xi64>) -> !pt {
  // CHECK: lwe.rlwe_encode
  // CHECK-NOT: level
  %pt = lwe.rlwe_encode %value {encoding = #full_crt_packing_encoding, ring = #ring_Z65537_i64_1_x32_} : tensor<32xi64> -> !pt
  return %pt : !pt
}

// CHECK: func @stale_level
func.func @stale_level(%value: tensor<32xi64>) -> !pt {
  // CHECK: lwe.rlwe_encode
  // CHECK-NOT: level
  %pt = lwe.rlwe_encode %value {encoding = #full_crt_packing_encoding, ring = #ring_Z65537_i64_1_x32_, level = 3 : i64} : tensor<32xi64> -> !pt
  return %pt : !pt
}

// A plaintext forwarded into a callee is left alone. The ciphertext the call
// happens to carry is not necessarily the one the plaintext is combined with
// inside the callee, and a level below a real use would silently truncate the
// ciphertext it meets there; falling back to the top of the chain only costs
// encoding work.
// CHECK: func @forwarded_to_call
func.func @forwarded_to_call(%ct: !ct_L1_, %value: tensor<32xi64>) -> !ct_L1_ {
  // CHECK: lwe.rlwe_encode
  // CHECK-NOT: level
  %pt = lwe.rlwe_encode %value {encoding = #full_crt_packing_encoding, ring = #ring_Z65537_i64_1_x32_} : tensor<32xi64> -> !pt
  %res = func.call @combine(%ct, %pt) : (!ct_L1_, !pt) -> !ct_L1_
  return %res : !ct_L1_
}
func.func private @combine(!ct_L1_, !pt) -> !ct_L1_

// Likewise for a plaintext carried through a region: the walk does not follow
// values into block arguments, so the ciphertext the loop carries alongside it
// says nothing about where the plaintext lands.
// CHECK: func @loop_carried
func.func @loop_carried(%ct: !ct_L1_, %value: tensor<32xi64>) -> !ct_L1_ {
  // CHECK: lwe.rlwe_encode
  // CHECK-NOT: level
  %pt = lwe.rlwe_encode %value {encoding = #full_crt_packing_encoding, ring = #ring_Z65537_i64_1_x32_} : tensor<32xi64> -> !pt
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %res:2 = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %ct, %p = %pt) -> (!ct_L1_, !pt) {
    %mul = lwe.rmul_plain %acc, %p : (!ct_L1_, !pt) -> !ct_L1_
    scf.yield %mul, %p : !ct_L1_, !pt
  }
  return %res#0 : !ct_L1_
}

// -----

// BFV keeps every ciphertext at the bottom of the modulus chain while the
// backend's ciphertexts span the whole chain, so `current` is not an encoding
// level and the encode op is left alone.

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain_L1_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 17179967489 : i64>, current = 0>
!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
#ring_Z65537_i64_1_x32_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**32>>
#ring_rns_L0_1_x32_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**32>>
#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x32_, encoding = #full_crt_packing_encoding>
#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x32_, encryption_type = lsb>
!pt = !lwe.lwe_plaintext<plaintext_space = #plaintext_space>
!ct_L0_ = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L1_C0_>

module attributes {scheme.bfv} {
  // CHECK: func @bfv
  func.func @bfv(%ct: !ct_L0_, %value: tensor<32xi64>) -> !ct_L0_ {
    // CHECK: lwe.rlwe_encode
    // CHECK-NOT: level
    %pt = lwe.rlwe_encode %value {encoding = #full_crt_packing_encoding, ring = #ring_Z65537_i64_1_x32_} : tensor<32xi64> -> !pt
    %res = lwe.rmul_plain %ct, %pt : (!ct_L0_, !pt) -> !ct_L0_
    return %res : !ct_L0_
  }
}

// -----

// `lwe.trivial_encrypt` fixes the level exactly, the same way `lwe.rlwe_encrypt`
// does: it has no ciphertext operand to be clamped against, so the backend
// builds the ciphertext at the plaintext's own level. The walk must not follow
// its ciphertext result either, or a lower-level ct-pt op downstream of that
// ciphertext would pull the plaintext below the level it was encrypted at.

!Z1005037682689_i64_ = !mod_arith.int<1005037682689 : i64>
!Z1032955396097_i64_ = !mod_arith.int<1032955396097 : i64>
!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z998595133441_i64_ = !mod_arith.int<998595133441 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain_L5_C1_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 1>
#modulus_chain_L5_C3_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 3>
!rns_L1_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_>
!rns_L3_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_, !Z1005037682689_i64_, !Z998595133441_i64_>
#ring_Z65537_i64_1_x32_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**32>>
#ring_rns_L1_1_x32_ = #polynomial.ring<coefficientType = !rns_L1_, polynomialModulus = <1 + x**32>>
#ring_rns_L3_1_x32_ = #polynomial.ring<coefficientType = !rns_L3_, polynomialModulus = <1 + x**32>>
#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x32_, encoding = #full_crt_packing_encoding>
#ciphertext_space_L1_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x32_, encryption_type = lsb>
#ciphertext_space_L3_ = #lwe.ciphertext_space<ring = #ring_rns_L3_1_x32_, encryption_type = lsb>
!pt = !lwe.lwe_plaintext<plaintext_space = #plaintext_space>
!ct_L1_ = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L5_C1_>
!ct_L3_ = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L3_, key = #key, modulus_chain = #modulus_chain_L5_C3_>

// CHECK: func @trivial_encrypt
func.func @trivial_encrypt(%value: tensor<32xi64>) -> !ct_L3_ {
  // CHECK: lwe.rlwe_encode
  // CHECK-SAME: level = 3 : i64
  %pt = lwe.rlwe_encode %value {encoding = #full_crt_packing_encoding, ring = #ring_Z65537_i64_1_x32_} : tensor<32xi64> -> !pt
  %ct = lwe.trivial_encrypt %pt : !pt -> !ct_L3_
  return %ct : !ct_L3_
}

// A ct-pt use below the trivial encryption's level is covered by it, exactly as
// for `lwe.rlwe_encrypt`.
// CHECK: func @trivial_encrypt_covers_combine
func.func @trivial_encrypt_covers_combine(%ct1: !ct_L1_, %value: tensor<32xi64>) -> (!ct_L3_, !ct_L1_) {
  // CHECK: lwe.rlwe_encode
  // CHECK-SAME: level = 3 : i64
  %pt = lwe.rlwe_encode %value {encoding = #full_crt_packing_encoding, ring = #ring_Z65537_i64_1_x32_} : tensor<32xi64> -> !pt
  %ct3 = lwe.trivial_encrypt %pt : !pt -> !ct_L3_
  %res = lwe.rmul_plain %ct1, %pt : (!ct_L1_, !pt) -> !ct_L1_
  return %ct3, %res : !ct_L3_, !ct_L1_
}

// The walk must stop at the trivial encryption rather than continue over its
// ciphertext result. Following it would reach the level-1 rmul_plain below and
// annotate level 1, which is a two-limb plaintext trivially encrypted into a
// four-limb ciphertext.
// CHECK: func @no_walk_through_ciphertext
func.func @no_walk_through_ciphertext(%value: tensor<32xi64>, %other: !pt) -> !ct_L1_ {
  // CHECK: lwe.rlwe_encode
  // CHECK-SAME: level = 3 : i64
  %pt = lwe.rlwe_encode %value {encoding = #full_crt_packing_encoding, ring = #ring_Z65537_i64_1_x32_} : tensor<32xi64> -> !pt
  %ct3 = lwe.trivial_encrypt %pt : !pt -> !ct_L3_
  %ct1 = bgv.modulus_switch %ct3 {to_ring = #ring_rns_L1_1_x32_} : !ct_L3_ -> !ct_L1_
  %res = lwe.rmul_plain %ct1, %other : (!ct_L1_, !pt) -> !ct_L1_
  return %res : !ct_L1_
}

// A user that turns the plaintext into something other than a plaintext is a
// consumer the walk does not model, so the fallback takes over. Following such a
// result would leave plaintext dataflow: here the walk would run decode ->
// cleartext -> re-encode -> rmul_plain and charge the first plaintext with the
// level of a ciphertext it never meets, even though its only use is the decode.
// CHECK: func @walk_stays_in_plaintext
func.func @walk_stays_in_plaintext(%ct1: !ct_L1_, %value: tensor<32xi64>) -> !ct_L1_ {
  // CHECK: lwe.rlwe_encode
  // CHECK-NOT: level
  %pt = lwe.rlwe_encode %value {encoding = #full_crt_packing_encoding, ring = #ring_Z65537_i64_1_x32_} : tensor<32xi64> -> !pt
  %decoded = lwe.rlwe_decode %pt {encoding = #full_crt_packing_encoding, ring = #ring_Z65537_i64_1_x32_} : !pt -> tensor<32xi64>
  // The re-encoded plaintext does reach the ciphertext, so it keeps its level.
  // CHECK: lwe.rlwe_encode
  // CHECK-SAME: level = 1 : i64
  %pt2 = lwe.rlwe_encode %decoded {encoding = #full_crt_packing_encoding, ring = #ring_Z65537_i64_1_x32_} : tensor<32xi64> -> !pt
  %res = lwe.rmul_plain %ct1, %pt2 : (!ct_L1_, !pt) -> !ct_L1_
  return %res : !ct_L1_
}
