// RUN: heir-opt --verify-diagnostics --split-input-file %s

// expected-error@below {{overflow must be either preserve_overflow or no_overflow, but found i1}}
#application = #lwe.application_data<message_type = i1, overflow = i1>

// -----

#poly = #polynomial.int_polynomial<x**1024 + 10>
#ring = #polynomial.ring<coefficientType=!mod_arith.int<65536:i32>, polynomialModulus=#poly>
#crt = #lwe.full_crt_packing_encoding<scaling_factor = 10000>
// expected-error@below {{polynomial modulus must be of the form x^n + 1}}
#plaintext_space = #lwe.plaintext_space<ring = #ring, encoding = #crt>

// -----

#poly = #polynomial.int_polynomial<x**1024 + 1>
#ring = #polynomial.ring<coefficientType=!mod_arith.int<12220:i32>, polynomialModulus=#poly>
#crt = #lwe.full_crt_packing_encoding<scaling_factor = 10000>
// expected-error@below {{modulus must be 1 mod n for full CRT packing}}
#plaintext_space = #lwe.plaintext_space<ring = #ring, encoding = #crt>

// -----

#key = #lwe.key<slot_index = 1>
#my_poly = #polynomial.int_polynomial<1 + x**1024>
#ring = #polynomial.ring<coefficientType = !mod_arith.int<12220:i32>, polynomialModulus=#my_poly>
!public_key = !lwe.lwe_public_key<key = #key, ring = #ring>

#preserve_overflow = #lwe.preserve_overflow<>
#application_data = #lwe.application_data<message_type = i1, overflow = #preserve_overflow>
#inverse_canonical_enc = #lwe.inverse_canonical_encoding<scaling_factor = 10000>
#plaintext_space = #lwe.plaintext_space<ring = #ring, encoding = #inverse_canonical_enc>
#ciphertext_space = #lwe.ciphertext_space<ring = #ring, encryption_type = msb, size = 3>
#modulus_chain = #lwe.modulus_chain<elements = <7917 : i32>, current = 0>

// expected-error@below {{a ciphertext with nontrivial slot rotation must have size 2, but found size 3}}
!lwe_ciphertext = !lwe.lwe_ciphertext<application_data = #application_data, plaintext_space = #plaintext_space, key = #key, ciphertext_space = #ciphertext_space, modulus_chain = #modulus_chain>

// -----

#key = #lwe.key<>
#my_poly = #polynomial.int_polynomial<1 + x**1024>
#ring = #polynomial.ring<coefficientType = !rns.rns<!mod_arith.int<12220:i32>>, polynomialModulus=#my_poly>
!public_key = !lwe.lwe_public_key<key = #key, ring = #ring>

#preserve_overflow = #lwe.preserve_overflow<>
#application_data = #lwe.application_data<message_type = i1, overflow = #preserve_overflow>
#inverse_canonical_enc = #lwe.inverse_canonical_encoding<scaling_factor = 10000>
#plaintext_space = #lwe.plaintext_space<ring = #ring, encoding = #inverse_canonical_enc>
#ciphertext_space = #lwe.ciphertext_space<ring = #ring, encryption_type = msb, size = 3>
#modulus_chain = #lwe.modulus_chain<elements = <7917 : i32, 65537 : i32>, current = 1>

// expected-error@below {{the level in the ciphertext ring must match the modulus chain's current, but found 0 and 1}}
!lwe_ciphertext = !lwe.lwe_ciphertext<application_data = #application_data, plaintext_space = #plaintext_space, key = #key, ciphertext_space = #ciphertext_space, modulus_chain = #modulus_chain>
