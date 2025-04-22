// RUN: heir-opt --verify-diagnostics --split-input-file %s

!Zp1 = !mod_arith.int<3721063133 : i64>
!Zp2 = !mod_arith.int<2737228591 : i64>
!Zp3 = !mod_arith.int<3180146689 : i64>

!ty_modarith = !rns.rns<!Zp1, !Zp2, !Zp3>

func.func @test_syntax_modarith(%arg0: !ty_modarith) -> !ty_modarith {
  return %arg0 : !ty_modarith
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
