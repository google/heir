// RUN: heir-opt --implement-trivial-encryption-as-addition %s | FileCheck %s

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z1032955396097_i64_ = !mod_arith.int<1032955396097 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>
!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
!rns_L1_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_>
#ring_Z65537_i64_1_x32_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**32>>
#ring_rns_L0_1_x32_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**32>>
#ring_rns_L1_1_x32_ = #polynomial.ring<coefficientType = !rns_L1_, polynomialModulus = <1 + x**32>>
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>
#modulus_chain_L5_C1_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 1>
#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x32_, encoding = #full_crt_packing_encoding>
#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x32_, encryption_type = lsb>
#ciphertext_space_L1_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x32_, encryption_type = lsb>
!ty = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>
!ty2 = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L5_C1_>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_Z65537_i64_1_x32_, encoding = #full_crt_packing_encoding>>
!pkey = !lwe.lwe_public_key<key = #key, ring = #ring_rns_L0_1_x32_>

module attributes {scheme.requested_slot_count = 32 : i64} {
  // CHECK: func @multiple_conceal
  // CHECK: lwe.rlwe_encode
  // CHECK: lwe.radd_plain
  // CHECK: lwe.radd_plain
  // CHECK: return

  func.func @multiple_conceal(%arg0: !ty, %cond: i1) -> (!ty, !ty) {
    %cst = arith.constant dense<4> : tensor<1024xi16>
    %pt = lwe.rlwe_encode %cst {
      encoding = #full_crt_packing_encoding,
      ring = #ring_Z65537_i64_1_x32_
    } : tensor<1024xi16> -> !pt
    %4 = lwe.trivial_encrypt %pt {ciphertext_bits = 64 : index} : !pt -> !ty
    %5 = lwe.trivial_encrypt %pt {ciphertext_bits = 64 : index} : !pt -> !ty
    return %4, %5 : !ty, !ty
  }

  func.func @enc_helper(%arg0: tensor<32xf32>, %pk: !pkey) attributes {client.enc_func = {func_name = "doctest", index = 0 : i64}} {
    return
  }
  // CHECK: func @multiple_conceal__encrypt__zero__0(
  // CHECK-NOT: func @multiple_conceal__encrypt__zero__1(

  // CHECK: func @multiple_conceal_different_types
  // CHECK: lwe.rlwe_encode
  // CHECK: lwe.radd_plain
  // CHECK: lwe.radd_plain
  // CHECK: return

  // CHECK-DAG: func @multiple_conceal_different_types__encrypt__zero__0(
  // CHECK-DAG: func @multiple_conceal_different_types__encrypt__zero__1(
  func.func @multiple_conceal_different_types(%arg0: !ty, %cond: i1) -> (!ty, !ty2) {
    %cst = arith.constant dense<4> : tensor<1024xi16>
    %pt = lwe.rlwe_encode %cst {
      encoding = #full_crt_packing_encoding,
      ring = #ring_Z65537_i64_1_x32_
    } : tensor<1024xi16> -> !pt
    %4 = lwe.trivial_encrypt %pt {ciphertext_bits = 64 : index} : !pt -> !ty
    %5 = lwe.trivial_encrypt %pt {ciphertext_bits = 64 : index} : !pt -> !ty2
    return %4, %5 : !ty, !ty2
  }

  // CHECK: func @multiple_conceal_different_mgmt
  // CHECK: lwe.rlwe_encode
  // CHECK: lwe.radd_plain
  // CHECK: lwe.radd_plain
  // CHECK: return

  // CHECK-DAG: func @multiple_conceal_different_mgmt__encrypt__zero__0(
  // CHECK-DAG: func @multiple_conceal_different_mgmt__encrypt__zero__1(
  func.func @multiple_conceal_different_mgmt(%arg0: !ty, %cond: i1) -> (!ty, !ty) {
    %cst = arith.constant dense<4> : tensor<1024xi16>
    %pt = lwe.rlwe_encode %cst {
      encoding = #full_crt_packing_encoding,
      ring = #ring_Z65537_i64_1_x32_
    } : tensor<1024xi16> -> !pt
    %4 = lwe.trivial_encrypt %pt {ciphertext_bits = 64 : index, mgmt.mgmt = #mgmt.mgmt<level = 0>} : !pt -> !ty
    %5 = lwe.trivial_encrypt %pt {ciphertext_bits = 64 : index, mgmt.mgmt = #mgmt.mgmt<level = 1>} : !pt -> !ty
    return %4, %5 : !ty, !ty
  }
}
