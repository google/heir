// RUN: heir-opt --lwe-add-client-interface --verify-diagnostics %s

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

!in_ty = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<32xi16>>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>
!out_ty = !lwe.new_lwe_ciphertext<application_data = <message_type = i16>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

#alignment = #tensor_ext.alignment<in = [32], out = [4096]>
#layout = #tensor_ext.layout<map = (d0) -> (d0), alignment = #alignment>
#original_type = #tensor_ext.original_type<originalType = tensor<32xi16>, layout = #layout>

#scalar_alignment = #tensor_ext.alignment<in = [], out = [1], insertedDims = [0]>
#scalar_layout = #tensor_ext.layout<map = (d0) -> (d0), alignment = #scalar_alignment>
#scalar_original_type = #tensor_ext.original_type<originalType = i16, layout = #scalar_layout>

// Missing the module attribute, which is required by add-client-interface to
// determine the slot count, though we could also use the ciphertext type's
// ciphertext space to get N.
// expected-error@+1 {{Unable to determine encryption type due to missing or unsupported scheme param attribute}}
module {
  func.func @simple_sum(
      %arg0: !in_ty {tensor_ext.original_type = #original_type}
  ) -> (!out_ty {tensor_ext.original_type = #scalar_original_type}) {
    %c31 = arith.constant 31 : index
    %0 = bgv.rotate_cols %arg0 { offset = 16 } : !in_ty
    %1 = bgv.add %arg0, %0 : (!in_ty, !in_ty) -> !in_ty
    %2 = bgv.rotate_cols %1 { offset = 8 } : !in_ty
    %3 = bgv.add %1, %2 : (!in_ty, !in_ty) -> !in_ty
    %4 = bgv.rotate_cols %3 { offset = 4 } : !in_ty
    %5 = bgv.add %3, %4 : (!in_ty, !in_ty) -> !in_ty
    %6 = bgv.rotate_cols %5 { offset = 2 } : !in_ty
    %7 = bgv.add %5, %6 : (!in_ty, !in_ty) -> !in_ty
    %8 = bgv.rotate_cols %7 { offset = 1 } : !in_ty
    %9 = bgv.add %7, %8 : (!in_ty, !in_ty) -> !in_ty
    %10 = bgv.extract %9, %c31 : (!in_ty, index) -> !out_ty
    return %10 : !out_ty
  }
}
