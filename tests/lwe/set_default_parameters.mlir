// RUN: heir-opt --lwe-set-default-parameters %s

#encoding = #lwe.bit_field_encoding<cleartext_start=30, cleartext_bitwidth=3>
#poly = #polynomial.int_polynomial<1 + x**1024>
!plaintext = !lwe.lwe_plaintext<encoding = #encoding>
!ciphertext = !lwe.lwe_ciphertext<encoding = #encoding>

// CHECK-LABEL: @test_adds_attribute
func.func @test_adds_attribute(%arg0 : !ciphertext) -> !ciphertext {
  %0 = arith.constant 0 : i1
  %1 = arith.constant 1 : i1
  %2 = lwe.encode %0 { encoding = #encoding }: i1 to !plaintext
  %3 = lwe.encode %1 { encoding = #encoding }: i1 to !plaintext
  // CHECK: lwe.trivial_encrypt
  // CHECK-SAME: lwe_params
  %4 = lwe.trivial_encrypt %2 : !plaintext to !ciphertext
  // CHECK: lwe.trivial_encrypt
  // CHECK-SAME: lwe_params
  %5 = lwe.trivial_encrypt %3 : !plaintext to !ciphertext
  // CHECK: lwe.lwe_ciphertext
  // CHECK-SAME: lwe_params
  %7 = cggi.lut3 %arg0, %4, %5 {lookup_table = 127 : index} : !ciphertext
  return %7 : !ciphertext
}
