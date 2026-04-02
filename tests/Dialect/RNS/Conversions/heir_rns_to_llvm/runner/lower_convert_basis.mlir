!Z3 = !mod_arith.int<3 : i64>
!Z5 = !mod_arith.int<5 : i64>
!Z7 = !mod_arith.int<7 : i64>


!rns3 = !rns.rns<!Z3>
!rns5 = !rns.rns<!Z5>
!rns7 = !rns.rns<!Z7>
!rns35 = !rns.rns<!Z3, !Z5>
!rns37 = !rns.rns<!Z3, !Z7>
!rns57 = !rns.rns<!Z5, !Z7>
!rns53 = !rns.rns<!Z5, !Z3>
!rns73 = !rns.rns<!Z7, !Z3>
!rns75 = !rns.rns<!Z7, !Z5>
!rns357 = !rns.rns<!Z3, !Z5, !Z7>


func.func public @test_convert_basis_5_35(%arg0: !rns5) -> !rns35 {
  %0 = rns.convert_basis %arg0 {targetBasis = !rns35} : !rns5 -> !rns35
  return %0 : !rns35
}

func.func public @test_convert_basis_5_57(%arg0: !rns5) -> !rns57 {
  %0 = rns.convert_basis %arg0 {targetBasis = !rns57} : !rns5 -> !rns57
  return %0 : !rns57
}

func.func public @test_convert_basis_7_37(%arg0: !rns7) -> !rns37 {
  %0 = rns.convert_basis %arg0 {targetBasis = !rns37} : !rns7 -> !rns37
  return %0 : !rns37
}

func.func public @test_convert_basis_7_57(%arg0: !rns7) -> !rns57 {
  %0 = rns.convert_basis %arg0 {targetBasis = !rns57} : !rns7 -> !rns57
  return %0 : !rns57
}

func.func public @test_convert_basis_3_357(%arg0: !rns3) -> !rns357 {
  %0 = rns.convert_basis %arg0 {targetBasis = !rns357} : !rns3 -> !rns357
  return %0 : !rns357
}

func.func public @test_convert_basis_5_357(%arg0: !rns5) -> !rns357 {
  %0 = rns.convert_basis %arg0 {targetBasis = !rns357} : !rns5 -> !rns357
  return %0 : !rns357
}

func.func public @test_convert_basis_7_357(%arg0: !rns7) -> !rns357 {
  %0 = rns.convert_basis %arg0 {targetBasis = !rns357} : !rns7 -> !rns357
  return %0 : !rns357
}

func.func public @test_convert_basis_35_357(%arg0: !rns35) -> !rns357 {
  %0 = rns.convert_basis %arg0 {targetBasis = !rns357} : !rns35 -> !rns357
  return %0 : !rns357
}

func.func public @test_convert_basis_37_357(%arg0: !rns37) -> !rns357 {
  %0 = rns.convert_basis %arg0 {targetBasis = !rns357} : !rns37 -> !rns357
  return %0 : !rns357
}

func.func public @test_convert_basis_57_357(%arg0: !rns57) -> !rns357 {
  %0 = rns.convert_basis %arg0 {targetBasis = !rns357} : !rns57 -> !rns357
  return %0 : !rns357
}

func.func public @test_convert_basis_53_357(%arg0: !rns53) -> !rns357 {
  %0 = rns.convert_basis %arg0 {targetBasis = !rns357} : !rns53 -> !rns357
  return %0 : !rns357
}

func.func public @test_convert_basis_73_357(%arg0: !rns73) -> !rns357 {
  %0 = rns.convert_basis %arg0 {targetBasis = !rns357} : !rns73 -> !rns357
  return %0 : !rns357
}

func.func public @test_convert_basis_75_357(%arg0: !rns75) -> !rns357 {
  %0 = rns.convert_basis %arg0 {targetBasis = !rns357} : !rns75 -> !rns357
  return %0 : !rns357
}
