module {
  // CHECK-LABEL: func @test_create_trivial_bool
  func.func @test_create_trivial_bool(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
    %c-128_i32 = arith.constant -128 : i32
    // %3 = arith.muli %arg0, %arg1 : i32
    %4 = arith.addi %arg2, %c-128_i32 : i32
    return %4 : i32
  }
}
