// RUN: heir-opt %s

// This simply tests for syntax.
#three_bit = #lwe.bit_field_encoding<cleartext_start=30, cleartext_bitwidth=3>
!cleartext = i3
!plaintext = i32
!ciphertext = tensor<1024x!plaintext,#three_bit>
module {
  func.func @test_syntax(%arg0 : !plaintext, %arg1 : !ciphertext) {
    %0 = cggi.lut3 (%arg1, %arg1, %arg1) {lookup_table = 127 : index} : !ciphertext
    %1 = cggi.scale (%arg1, %arg0) : (!ciphertext, !plaintext) -> !ciphertext
    return
  }
}
