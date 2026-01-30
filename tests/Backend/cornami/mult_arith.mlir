func.func @test_int_mul(%lhs : i8 {secret.secret}, %rhs : i8 {secret.secret}) -> i8 {
  %res = arith.muli %lhs, %rhs : i8
  return %res : i8
}
