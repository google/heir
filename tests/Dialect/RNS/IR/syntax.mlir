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

func.func @test_extract_residue(%arg0: !ty_modarith) -> !Zp2 {
  %0 = rns.extract_residue %arg0 {index = 1 : index} : !ty_modarith -> !Zp2
  return %0 : !Zp2
}

func.func @elementwise_extract_residue(%arg0: tensor<10x!ty_modarith>) -> tensor<10x!Zp2> {
  %0 = rns.extract_residue %arg0 {index = 1 : index} : tensor<10x!ty_modarith> -> tensor<10x!Zp2>
  return %0 : tensor<10x!Zp2>
}

func.func @test_pack(%arg0: !Zp1, %arg1: !Zp2) -> !ty_truncated {
  %0 = rns.pack %arg0, %arg1 : !Zp1, !Zp2
  return %0 : !ty_truncated
}

func.func @test_convert_basis(%arg0: !ty_truncated) -> !ty_modarith {
  %0 = rns.convert_basis %arg0 {targetBasis = !ty_modarith} : !ty_truncated -> !ty_modarith
  return %0 : !ty_modarith
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

// -----

!Zp1_verify = !mod_arith.int<3721063133 : i64>
!Zp2_verify = !mod_arith.int<2737228591 : i64>
!Zp3_verify = !mod_arith.int<3180146689 : i64>
!ty_modarith_verify = !rns.rns<!Zp1_verify, !Zp2_verify, !Zp3_verify>

func.func @test_extract_residue_verifier_oob_index(%arg0: !ty_modarith_verify) {
  // expected-error@+1 {{'rns.extract_residue' index 3 is out of bounds for an RNS type with 3 limbs}}
  %0 = rns.extract_residue %arg0 {index = 3 : index} : !ty_modarith_verify -> !Zp1_verify
  return
}
