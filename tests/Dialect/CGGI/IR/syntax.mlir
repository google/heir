// RUN: heir-opt %s | FileCheck %s

// This simply tests for syntax.
#poly = #polynomial.int_polynomial<x>

#key = #lwe.key<slot_index = 0>
#pspace = #lwe.plaintext_space<ring = #polynomial.ring<coefficientType = i4, polynomialModulus = #poly>, encoding = #lwe.constant_coefficient_encoding<scaling_factor = 268435456>>
#cspace = #lwe.ciphertext_space<ring = #polynomial.ring<coefficientType = i32, polynomialModulus = #poly>, encryption_type = msb, size = 742>
!plaintext = !lwe.lwe_plaintext<plaintext_space = #pspace>
!ciphertext = !lwe.lwe_ciphertext<plaintext_space = #pspace, ciphertext_space = #cspace, key = #key>

module {
  //CHECK: test_syntax
  func.func @test_syntax(%arg0 : !ciphertext) -> !ciphertext {
    %0 = arith.constant 0 : i1
    %1 = arith.constant 1 : i1
    %2 = lwe.encode %0 { plaintext_bits = 4 : index } : i1 to !plaintext
    %3 = lwe.encode %1 { plaintext_bits = 4 : index } : i1 to !plaintext
    %4 = lwe.trivial_encrypt %2 { ciphertext_bits = 32 : index }: !plaintext to !ciphertext
    %5 = lwe.trivial_encrypt %3 { ciphertext_bits = 32 : index }: !plaintext to !ciphertext
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
