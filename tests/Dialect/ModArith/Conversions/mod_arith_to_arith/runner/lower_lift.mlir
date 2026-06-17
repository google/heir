!Zp3 = !mod_arith.int<3 : i32>
!Zp4 = !mod_arith.int<4 : i32>
!TZ3 = tensor<3x!Zp3>
!TZ4 = tensor<4x!Zp4>

func.func @test_lower_lift_centered() -> tensor<14xi32> {
  %x3 = arith.constant dense<[0, 1, 2]> : tensor<3xi32>
  %x3c = arith.constant dense<[0, 1, -1]> : tensor<3xi32>
  %x4 = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  %x4c = arith.constant dense<[0, 1, -2, -1]> : tensor<4xi32>

  %ex3 = mod_arith.encapsulate %x3 : tensor<3xi32> -> !TZ3
  %ex3c = mod_arith.encapsulate %x3c : tensor<3xi32> -> !TZ3
  %ex4 = mod_arith.encapsulate %x4 : tensor<4xi32> -> !TZ4
  %ex4c = mod_arith.encapsulate %x4c : tensor<4xi32> -> !TZ4

  %lx3 = mod_arith.lift centered %ex3 : !TZ3 -> tensor<3xi32>
  %lx3c = mod_arith.lift centered %ex3c : !TZ3 -> tensor<3xi32>
  %lx4 = mod_arith.lift centered %ex4 : !TZ4 -> tensor<4xi32>
  %lx4c = mod_arith.lift centered %ex4c : !TZ4 -> tensor<4xi32>

  %t1 = tensor.concat dim(0) %lx3, %lx3c : (tensor<3xi32>, tensor<3xi32>) -> tensor<6xi32>
  %t2 = tensor.concat dim(0) %t1, %lx4 : (tensor<6xi32>, tensor<4xi32>) -> tensor<10xi32>
  %result = tensor.concat dim(0) %t2, %lx4c : (tensor<10xi32>, tensor<4xi32>) -> tensor<14xi32>
  return %result : tensor<14xi32>
}

func.func @test_lower_lift_standard() -> tensor<14xi32> {
  %x3 = arith.constant dense<[0, 1, 2]> : tensor<3xi32>
  %x3c = arith.constant dense<[0, 1, -1]> : tensor<3xi32>
  %x4 = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  %x4c = arith.constant dense<[0, 1, -2, -1]> : tensor<4xi32>

  %ex3 = mod_arith.encapsulate %x3 : tensor<3xi32> -> !TZ3
  %ex3c = mod_arith.encapsulate %x3c : tensor<3xi32> -> !TZ3
  %ex4 = mod_arith.encapsulate %x4 : tensor<4xi32> -> !TZ4
  %ex4c = mod_arith.encapsulate %x4c : tensor<4xi32> -> !TZ4

  %lx3 = mod_arith.lift standard %ex3 : !TZ3 -> tensor<3xi32>
  %lx3c = mod_arith.lift standard %ex3c : !TZ3 -> tensor<3xi32>
  %lx4 = mod_arith.lift standard %ex4 : !TZ4 -> tensor<4xi32>
  %lx4c = mod_arith.lift standard %ex4c : !TZ4 -> tensor<4xi32>

  %t1 = tensor.concat dim(0) %lx3, %lx3c : (tensor<3xi32>, tensor<3xi32>) -> tensor<6xi32>
  %t2 = tensor.concat dim(0) %t1, %lx4 : (tensor<6xi32>, tensor<4xi32>) -> tensor<10xi32>
  %result = tensor.concat dim(0) %t2, %lx4c : (tensor<10xi32>, tensor<4xi32>) -> tensor<14xi32>
  return %result : tensor<14xi32>
}
