// RUN: heir-opt %s | FileCheck %s

// This simply tests for syntax.
#encoding = #lwe.bit_field_encoding<cleartext_start=30, cleartext_bitwidth=3>
#poly = #polynomial.int_polynomial<1 + x**1024>
#params = #lwe.lwe_params<cmod=7917, dimension=4>
!plaintext = !lwe.lwe_plaintext<encoding = #encoding>
!ciphertext = !lwe.lwe_ciphertext<encoding = #encoding, lwe_params = #params>

module {
  //CHECK-LABEL: test_syntax
  func.func @test_syntax(%arg0 : !ciphertext) -> !ciphertext {
    %0 = arith.constant 0 : i1
    %1 = arith.constant 1 : i1
    %2 = lwe.encode %0 { encoding = #encoding }: i1 to !plaintext
    %3 = lwe.encode %1 { encoding = #encoding }: i1 to !plaintext
    %4 = lwe.trivial_encrypt %2 { params = #params } : !plaintext to !ciphertext
    %5 = lwe.trivial_encrypt %3 { params = #params } : !plaintext to !ciphertext
    %6 = cggi.lut3 %arg0, %4, %5 {lookup_table = 127 : index} : !ciphertext
    %c3 = arith.constant 3 : i3
    %7 = lwe.mul_scalar %4, %c3 : (!ciphertext, i3) -> !ciphertext
    %8 = lwe.add %7, %5 : !ciphertext
    %9 = cggi.lut_lincomb %4, %5, %6, %7 {coefficients = array<i32: 1, 1, 1, 2>, lookup_table = 68 : index} : !ciphertext

    %10, %11, %12, %13, %14 = cggi.multi_lut_lincomb %4, %5, %6, %7 {
      coefficients = array<i32: 1, 1, 1, 2>,
      lookup_tables = array<i32: 68, 70, 4, 8, 1>
    } : (!ciphertext, !ciphertext, !ciphertext, !ciphertext) -> (!ciphertext, !ciphertext, !ciphertext, !ciphertext, !ciphertext)

    return %14 : !ciphertext
  }
}
