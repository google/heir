// RUN: heir-opt %s --lwe-add-debug-port --lwe-to-cheddar | FileCheck %s

// With --debug, `--lwe-add-debug-port` lowers each `debug.validate` to a call to
// an external `@__heir_debug_N(secret_key, ciphertext)`. LWEToCheddar reshapes
// both the declaration and the call to `(encoder, user_interface, ciphertext)`
// so the CHEDDAR-side hook has everything it needs to decrypt (UserInterface)
// and decode (Encoder) the ciphertext for printing.

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 1099511627776>
#key = #lwe.key<>
#modulus_chain_L1_C1 = #lwe.modulus_chain<elements = <36028797019488257 : i64, 1099512938497 : i64>, current = 1>
#ring_f64_1_x65536 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**65536>>
!rns_L1 = !rns.rns<!mod_arith.int<36028797019488257 : i64>, !mod_arith.int<1099512938497 : i64>>
#ring_rns_L1_1_x65536 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**65536>>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x65536, encryption_type = mix>
!ct_L1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L1_C1>

module attributes {scheme.ckks, ckks.schemeParam = #ckks.scheme_param<logN = 16, Q = [36028797019488257, 1099512938497], P = [2305843009211596801], logDefaultScale = 40>} {
  // The reshaped external declaration: (encoder, user_interface, ciphertext).
  // heir-opt prints the cheddar types via auto-generated aliases !encoder etc.
  // CHECK: func.func private @__heir_debug_0
  // CHECK-SAME: !encoder
  // CHECK-SAME: !user_interface
  // CHECK-SAME: tensor<!ciphertext>

  // CHECK: func.func @debug
  // CHECK-SAME: !user_interface
  // CHECK-NOT: func.func @debug({{.*}}!user_interface{{.*}}!user_interface
  func.func @debug(%ct: !ct_L1) -> !ct_L1 {
    // The call threads the contextual encoder + user_interface, then the
    // ciphertext; the original secret-key operand is dropped.
    // CHECK: call @__heir_debug_0
    // CHECK-SAME: debug.name = "val0"
    %ct_0 = lwe.radd %ct, %ct : (!ct_L1, !ct_L1) -> !ct_L1
    debug.validate %ct_0 <name = "val0"> : !ct_L1
    return %ct_0 : !ct_L1
  }
}
