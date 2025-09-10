// RUN: heir-opt --verify-diagnostics --split-input-file %s

!Zp1 = !mod_arith.int<3721063133 : i64>
!Zp2 = !mod_arith.int<2737228591 : i64>
!Zp3 = !mod_arith.int<3180146689 : i64>

!ty_modarith = !rns.rns<!Zp1, !Zp2, !Zp3>
!ty_truncated = !rns.rns<!Zp1, !Zp2>

func.func @test_syntax_modarith(%arg0: !ty_modarith) -> !ty_modarith {
  %0 = rns.extract_slice %arg0 {start = 0 : index, size = 2 : index} : !ty_modarith -> !ty_truncated
  return %arg0 : !ty_modarith
}

func.func @elementwise_extract_slice(%arg0: tensor<10x!ty_modarith>) -> tensor<10x!ty_truncated> {
  %0 = rns.extract_slice %arg0 {start = 0 : index, size = 2 : index} : tensor<10x!ty_modarith> -> tensor<10x!ty_truncated>
  return %0 : tensor<10x!ty_truncated>
}

// expected-error@+1 {{RNS type has incompatible basis types}}
!ty_modarith_bad = !rns.rns<!Zp1, !Zp2, !Zp1>

// -----

!Zp1 = !mod_arith.int<3721063133 : i64>
!Zp2 = !mod_arith.int<65537 : i64>
!Zp2_i32 = !mod_arith.int<65537 : i32>

!ty_modarith = !rns.rns<!Zp1, !Zp2>

func.func @test_syntax_modarith(%arg0: !ty_modarith) -> !ty_modarith {
  return %arg0 : !ty_modarith
}

// expected-error@+1 {{RNS type has incompatible basis types}}
!ty_modarith_bad = !rns.rns<!Zp1, !Zp2_i32>

// -----

// expected-error@+1 {{does not have RNSBasisTypeInterface}}
!ty_int_bad = !rns.rns<i32, i64>

// -----

!Zp1_verify = !mod_arith.int<3721063133 : i64>
!Zp2_verify = !mod_arith.int<2737228591 : i64>
!Zp3_verify = !mod_arith.int<3180146689 : i64>
!ty_modarith_verify = !rns.rns<!Zp1_verify, !Zp2_verify, !Zp3_verify>
!ty_truncated_verify = !rns.rns<!Zp1_verify, !Zp2_verify>

func.func @test_extract_slice_verifier_negative_start(%arg0: !ty_modarith_verify) {
  // expected-error@+1 {{start index -1 cannot be negative}}
  %0 = rns.extract_slice %arg0 {start = -1 : index, size = 2 : index} : !ty_modarith_verify -> !ty_truncated_verify
  return
}

// -----

!Zp1_verify = !mod_arith.int<3721063133 : i64>
!Zp2_verify = !mod_arith.int<2737228591 : i64>
!Zp3_verify = !mod_arith.int<3180146689 : i64>
!ty_modarith_verify = !rns.rns<!Zp1_verify, !Zp2_verify, !Zp3_verify>
!ty_truncated_verify = !rns.rns<!Zp1_verify, !Zp2_verify>

func.func @test_extract_slice_verifier_negative_size(%arg0: !ty_modarith_verify) {
  // expected-error@+1 {{size -1 cannot be negative}}
  %0 = rns.extract_slice %arg0 {start = 0 : index, size = -1 : index} : !ty_modarith_verify -> !ty_truncated_verify
  return
}

// -----

!Zp1_verify = !mod_arith.int<3721063133 : i64>
!Zp2_verify = !mod_arith.int<2737228591 : i64>
!Zp3_verify = !mod_arith.int<3180146689 : i64>
!ty_modarith_verify = !rns.rns<!Zp1_verify, !Zp2_verify, !Zp3_verify>
!ty_truncated_verify = !rns.rns<!Zp1_verify, !Zp2_verify>

func.func @test_extract_slice_verifier_oob_start_plus_size(%arg0: !ty_modarith_verify) {
  // expected-error@+1 {{slice of size 3 starting at 1 is out of bounds for RNS type with 3 limbs}}
  %0 = rns.extract_slice %arg0 {start = 1 : index, size = 3 : index} : !ty_modarith_verify -> !ty_truncated_verify
  return
}

// -----

!Zp1_verify = !mod_arith.int<3721063133 : i64>
!Zp2_verify = !mod_arith.int<2737228591 : i64>
!Zp3_verify = !mod_arith.int<3180146689 : i64>
!ty_modarith_verify = !rns.rns<!Zp1_verify, !Zp2_verify, !Zp3_verify>
!ty_truncated_verify = !rns.rns<!Zp1_verify, !Zp2_verify>

func.func @test_extract_slice_verifier_oob_size(%arg0: !ty_modarith_verify) {
  // expected-error@+1 {{slice of size 4 starting at 0 is out of bounds for RNS type with 3 limbs}}
  %0 = rns.extract_slice %arg0 {start = 0 : index, size = 4 : index} : !ty_modarith_verify -> !ty_truncated_verify
  return
}

// -----

!Zp1_verify = !mod_arith.int<3721063133 : i64>
!Zp2_verify = !mod_arith.int<2737228591 : i64>
!Zp3_verify = !mod_arith.int<3180146689 : i64>
!ty_modarith_verify = !rns.rns<!Zp1_verify, !Zp2_verify, !Zp3_verify>
!ty_truncated_verify = !rns.rns<!Zp1_verify, !Zp2_verify>

func.func @test_extract_slice_verifier_oob_start(%arg0: !ty_modarith_verify) {
  // expected-error@+1 {{slice of size 1 starting at 3 is out of bounds for RNS type with 3 limbs}}
  %0 = rns.extract_slice %arg0 {start = 3 : index, size = 1 : index} : !ty_modarith_verify -> !ty_truncated_verify
  return
}
