#preserve_overflow = #lwe.preserve_overflow<>
#key = #lwe.key<slot_index = 0>
#poly = #polynomial.int_polynomial<1 + x**1024>
#pspace = #lwe.plaintext_space<
  ring = #polynomial.ring<coefficientType = i3, polynomialModulus = #poly>,
  encoding = #lwe.constant_coefficient_encoding<scaling_factor = 268435456>
>
#cspace = #lwe.ciphertext_space<
  ring = #polynomial.ring<coefficientType = i32, polynomialModulus = #poly>,
  encryption_type = msb,
  size = 2
>
!ct_ty = !lwe.lwe_ciphertext<
  application_data = <message_type = i3, overflow = #preserve_overflow>,
  plaintext_space = #pspace,
  ciphertext_space = #cspace,
  key = #key
>

func.func @full_adder_sum(%a: !ct_ty, %b: !ct_ty, %cin: !ct_ty) -> !ct_ty {
  %sum = cggi.lut_lincomb %a, %b, %cin {
    coefficients = array<i32: 1, 2, 4>,
    // For 3-input XOR (odd parity), LSb-first bit positions set at {1,2,4,7} => 150
    lookup_table = 150 : index
  } : !ct_ty
  return %sum : !ct_ty
}
