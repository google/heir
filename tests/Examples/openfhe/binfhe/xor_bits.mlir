// A simple XOR operation on two encrypted bits using CGGI/BinFHE

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

// XOR two encrypted bits
func.func @xor_bits(%arg0: !ct_ty, %arg1: !ct_ty) -> !ct_ty {
  // XOR truth table: 00->0, 01->1, 10->1, 11->0
  // Encoding with coefficients [1, 2] gives inputs: 0, 1, 2, 3
  // Lookup table for XOR: 0110 (binary) = 6 (decimal)
  %0 = cggi.lut_lincomb %arg0, %arg1 {
    coefficients = array<i32: 1, 2>,
    lookup_table = 6 : index
  } : !ct_ty
  return %0 : !ct_ty
}

// AND two encrypted bits
func.func @and_bits(%arg0: !ct_ty, %arg1: !ct_ty) -> !ct_ty {
  // AND truth table: 00->0, 01->0, 10->0, 11->1
  // Lookup table for AND: 1000 (binary) = 8 (decimal)
  %0 = cggi.lut_lincomb %arg0, %arg1 {
    coefficients = array<i32: 1, 2>,
    lookup_table = 8 : index
  } : !ct_ty
  return %0 : !ct_ty
}

// OR two encrypted bits
func.func @or_bits(%arg0: !ct_ty, %arg1: !ct_ty) -> !ct_ty {
  // OR truth table: 00->0, 01->1, 10->1, 11->1
  // Lookup table for OR: 1110 (binary) = 14 (decimal)
  %0 = cggi.lut_lincomb %arg0, %arg1 {
    coefficients = array<i32: 1, 2>,
    lookup_table = 14 : index
  } : !ct_ty
  return %0 : !ct_ty
}

// NOT an encrypted bit
func.func @not_bit(%arg0: !ct_ty) -> !ct_ty {
  // NOT truth table: 0->1, 1->0
  // Lookup table for NOT: 01 (binary) = 1 (decimal)
  %0 = cggi.lut_lincomb %arg0 {
    coefficients = array<i32: 1>,
    lookup_table = 1 : index
  } : !ct_ty
  return %0 : !ct_ty
}

// Full adder sum: compute sum for two bits and carry-in
func.func @full_adder_sum(%a: !ct_ty, %b: !ct_ty, %cin: !ct_ty) -> !ct_ty {
  // Sum = a XOR b XOR cin
  // Truth table for 3-input XOR: 0,1,1,0,1,0,0,1
  // With coefficients [1,2,4]: inputs map to 0-7
  // Lookup table: indices {1,2,4,7} LSb-first => 150 (decimal)
  %sum = cggi.lut_lincomb %a, %b, %cin {
    coefficients = array<i32: 1, 2, 4>,
    lookup_table = 150 : index
  } : !ct_ty

  return %sum : !ct_ty
}

// Full adder carry: compute carry for two bits and carry-in
func.func @full_adder_carry(%a: !ct_ty, %b: !ct_ty, %cin: !ct_ty) -> !ct_ty {
  // Carry = (a AND b) OR (cin AND (a XOR b))
  // Truth table: 0,0,0,1,0,1,1,1
  // Lookup table: 11101000 (binary) = 232 (decimal)
  %carry = cggi.lut_lincomb %a, %b, %cin {
    coefficients = array<i32: 1, 2, 4>,
    lookup_table = 232 : index
  } : !ct_ty

  return %carry : !ct_ty
}
