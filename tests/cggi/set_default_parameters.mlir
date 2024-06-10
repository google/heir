// RUN: heir-opt --cggi-set-default-parameters %s

#encoding = #lwe.bit_field_encoding<cleartext_start=30, cleartext_bitwidth=3>
#poly = #polynomial.int_polynomial<1 + x**1024>
#params = #lwe.lwe_params<cmod=7917, dimension=4>
!plaintext = !lwe.lwe_plaintext<encoding = #encoding>
!ciphertext = !lwe.lwe_ciphertext<encoding = #encoding, lwe_params = #params>

// CHECK-LABEL: @test_adds_attribute
func.func @test_adds_attribute(%arg0 : !ciphertext) -> !ciphertext {
  %0 = arith.constant 0 : i1
  %1 = arith.constant 1 : i1
  %2 = lwe.encode %0 { encoding = #encoding }: i1 to !plaintext
  %3 = lwe.encode %1 { encoding = #encoding }: i1 to !plaintext
  // CHECK: lwe.trivial_encrypt
  %4 = lwe.trivial_encrypt %2 { params = #params } : !plaintext to !ciphertext
  // CHECK: lwe.trivial_encrypt
  %5 = lwe.trivial_encrypt %3 { params = #params } : !plaintext to !ciphertext
  // CHECK: cggi.lut3
  // Testing that the pass adds the right kind of attribute to the op
  // CHECK: cggi_params = #cggi.cggi_params
  %6 = cggi.lut3 %arg0, %4, %5 {lookup_table = 127 : index} : !ciphertext
  return %4 : !ciphertext
}
