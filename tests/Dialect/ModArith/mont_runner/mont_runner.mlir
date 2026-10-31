!Zp1 = !mod_arith.int<974258177 : i32>
!Zp1_vec = tensor<2048x!Zp1>

!Zp2 = !mod_arith.int<1081212929 : i32>
!Zp2_vec = tensor<2048x!Zp2>

!Zp3 = !mod_arith.int<1125899908022273 : i64>
!Zp3_vec = tensor<2048x!Zp3>

// Prime 1 (32-bit: 974258177)
func.func public @test_mont_mul_p1(%a: !Zp1_vec, %b: !Zp1_vec) -> !Zp1_vec {
  %0 = mod_arith.mont_mul %a, %b : !Zp1_vec
  return %0 : !Zp1_vec
}

func.func public @test_mont_mul_lazy_p1(%a: !Zp1_vec, %b: !Zp1_vec) -> !Zp1_vec {
  %0 = mod_arith.mont_mul %a, %b lazy : !Zp1_vec
  return %0 : !Zp1_vec
}

func.func public @test_mont_reduce_p1(%w: tensor<2048xi64>) -> !Zp1_vec {
  %0 = mod_arith.mont_reduce %w : tensor<2048xi64> -> !Zp1_vec
  return %0 : !Zp1_vec
}

func.func public @test_mont_reduce_lazy_p1(%w: tensor<2048xi64>) -> !Zp1_vec {
  %0 = mod_arith.mont_reduce %w lazy : tensor<2048xi64> -> !Zp1_vec
  return %0 : !Zp1_vec
}

func.func public @test_to_mont_p1(%a: !Zp1_vec) -> !Zp1_vec {
  %0 = mod_arith.to_mont %a : !Zp1_vec
  return %0 : !Zp1_vec
}

func.func public @test_from_mont_p1(%a: !Zp1_vec) -> !Zp1_vec {
  %0 = mod_arith.from_mont %a : !Zp1_vec
  return %0 : !Zp1_vec
}

// Prime 2 (32-bit: 1081212929)
func.func public @test_mont_mul_p2(%a: !Zp2_vec, %b: !Zp2_vec) -> !Zp2_vec {
  %0 = mod_arith.mont_mul %a, %b : !Zp2_vec
  return %0 : !Zp2_vec
}

func.func public @test_mont_mul_lazy_p2(%a: !Zp2_vec, %b: !Zp2_vec) -> !Zp2_vec {
  %0 = mod_arith.mont_mul %a, %b lazy : !Zp2_vec
  return %0 : !Zp2_vec
}

func.func public @test_mont_reduce_p2(%w: tensor<2048xi64>) -> !Zp2_vec {
  %0 = mod_arith.mont_reduce %w : tensor<2048xi64> -> !Zp2_vec
  return %0 : !Zp2_vec
}

func.func public @test_mont_reduce_lazy_p2(%w: tensor<2048xi64>) -> !Zp2_vec {
  %0 = mod_arith.mont_reduce %w lazy : tensor<2048xi64> -> !Zp2_vec
  return %0 : !Zp2_vec
}

func.func public @test_to_mont_p2(%a: !Zp2_vec) -> !Zp2_vec {
  %0 = mod_arith.to_mont %a : !Zp2_vec
  return %0 : !Zp2_vec
}

func.func public @test_from_mont_p2(%a: !Zp2_vec) -> !Zp2_vec {
  %0 = mod_arith.from_mont %a : !Zp2_vec
  return %0 : !Zp2_vec
}

// Prime 3 (64-bit: 1125899908022273)
func.func public @test_mont_mul_p3(%a: !Zp3_vec, %b: !Zp3_vec) -> !Zp3_vec {
  %0 = mod_arith.mont_mul %a, %b : !Zp3_vec
  return %0 : !Zp3_vec
}

func.func public @test_mont_mul_lazy_p3(%a: !Zp3_vec, %b: !Zp3_vec) -> !Zp3_vec {
  %0 = mod_arith.mont_mul %a, %b lazy : !Zp3_vec
  return %0 : !Zp3_vec
}

func.func public @test_mont_reduce_p3(%w: tensor<2048xi128>) -> !Zp3_vec {
  %0 = mod_arith.mont_reduce %w : tensor<2048xi128> -> !Zp3_vec
  return %0 : !Zp3_vec
}

func.func public @test_mont_reduce_lazy_p3(%w: tensor<2048xi128>) -> !Zp3_vec {
  %0 = mod_arith.mont_reduce %w lazy : tensor<2048xi128> -> !Zp3_vec
  return %0 : !Zp3_vec
}

func.func public @test_to_mont_p3(%a: !Zp3_vec) -> !Zp3_vec {
  %0 = mod_arith.to_mont %a : !Zp3_vec
  return %0 : !Zp3_vec
}

func.func public @test_from_mont_p3(%a: !Zp3_vec) -> !Zp3_vec {
  %0 = mod_arith.from_mont %a : !Zp3_vec
  return %0 : !Zp3_vec
}
