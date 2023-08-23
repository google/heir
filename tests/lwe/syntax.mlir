// RUN: heir-opt %s > %t

// This simply tests for syntax.

#lwe_noise = #lwe.gaussian<domain_bitwidth=32, stdev=16384>
#encoding = #lwe.encoding_scheme<
  plaintext_bitwidth=32,
  padding_bitwidth=1,
  cleartext_bitwidth=3>
#mod_switched_encoding = #lwe.encoding_scheme<
  plaintext_bitwidth=16,
  padding_bitwidth=1,
  cleartext_bitwidth=3>
module {
  func.func @test_syntax(
      %arg0 : !lwe.ciphertext<encoding_scheme=#encoding, dimension=1024, secret_key=0>) {
    %c_4 = arith.constant 4 : i32
    %cleartext = arith.constant 6 : i3
    %plaintext = lwe.encode(%cleartext) : i3 -> !lwe.plaintext<encoding_scheme=#encoding>
    %arg1 = lwe.encrypt(%plaintext) : !lwe.plaintext<encoding_scheme=#encoding> -> !lwe.ciphertext<encoding_scheme=#encoding, dimension=1024, secret_key=0>
    %0 = lwe.add(%arg0, %arg1) : !lwe.ciphertext<encoding_scheme=#encoding, dimension=1024, secret_key=0>
    %1 = lwe.sub(%arg0, %arg1) : !lwe.ciphertext<encoding_scheme=#encoding, dimension=1024, secret_key=0>
    %2 = lwe.negate(%arg0) : !lwe.ciphertext<encoding_scheme=#encoding, dimension=1024, secret_key=0>
    %3 = lwe.scale(%arg0, %c_4) : (!lwe.ciphertext<encoding_scheme=#encoding, dimension=1024, secret_key=0>, i32) -> !lwe.ciphertext<encoding_scheme=#encoding, dimension=1024, secret_key=0>
    %4 = lwe.modulus_switch(%arg0) {from_log_modulus = 32 : index, to_log_modulus = 16 : index} : !lwe.ciphertext<encoding_scheme=#encoding, dimension=1024, secret_key=0> -> !lwe.ciphertext<encoding_scheme=#mod_switched_encoding, dimension=1024, secret_key=0>
    %5 = lwe.key_switch(%arg0) : !lwe.ciphertext<encoding_scheme=#encoding, dimension=1024, secret_key=0> -> !lwe.ciphertext<encoding_scheme=#encoding, dimension=512, secret_key=1>
    %6 = lwe.decrypt(%1) : !lwe.ciphertext<encoding_scheme=#encoding, dimension=1024, secret_key=0> -> !lwe.plaintext<encoding_scheme=#encoding>
    %7 = lwe.decode(%6) : !lwe.plaintext<encoding_scheme=#encoding> -> i3
    return
  }
}
