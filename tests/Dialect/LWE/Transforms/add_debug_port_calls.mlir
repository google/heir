// RUN: heir-opt --mlir-print-local-scope --lwe-add-debug-port %s | FileCheck %s

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>

!rns_L0_ = !rns.rns<!Z1095233372161_i64_>

#ring_Z65537_i64_1_x32_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**32>>
#ring_rns_L0_1_x32_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**32>>

#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>

#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>

#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x32_, encoding = #full_crt_packing_encoding>

#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x32_, encryption_type = lsb>

!ty = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

module {
  // CHECK-LABEL: func @callee
  // CHECK-SAME: (%[[SK:[^:]*]]: {{.*}}, %[[ARG:[^:]*]]: !lwe.lwe_ciphertext<{{.*}}>)
  func.func @callee(%arg0: !ty) {
    debug.validate %arg0 {name = "callee_debug"} : !ty
    return
  }

  // CHECK-LABEL: func @caller
  // CHECK-SAME: (%[[SK2:[^:]*]]: {{.*}}, %[[ARG2:[^:]*]]: !lwe.lwe_ciphertext<{{.*}}>)
  func.func @caller(%arg0: !ty) {
    // CHECK: call @callee(%[[SK2]], %[[ARG2]])
    func.call @callee(%arg0) : (!ty) -> ()
    return
  }
}
