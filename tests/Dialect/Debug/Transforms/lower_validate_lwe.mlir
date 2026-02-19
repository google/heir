// RUN: heir-opt --mlir-print-local-scope --lwe-add-debug-port %s | FileCheck %s

#ring_Z65537_i64_1_x32_ = #polynomial.ring<coefficientType = i64, polynomialModulus = <1 + x**32>>
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain = #lwe.modulus_chain<elements = <1095233372161 : i64>, current = 0>
#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x32_, encoding = #full_crt_packing_encoding>
#ciphertext_space = #lwe.ciphertext_space<ring = #ring_Z65537_i64_1_x32_, encryption_type = lsb>

!ct_ty = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space, key = #key, modulus_chain = #modulus_chain>

module {
  // CHECK: func.func private @__heir_debug_0
  func.func @test_lower_validate_lwe(%arg0: !ct_ty) -> !ct_ty {
    // CHECK-LABEL: func.func @test_lower_validate_lwe
    // CHECK: call @__heir_debug_0
    debug.validate %arg0 {name = "lwe_val1", metadata = "lwe_meta1"} : !ct_ty
    return %arg0 : !ct_ty
  }
}
