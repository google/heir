!Zp1 = !mod_arith.int<7681 : i26>
!Zp1v = tensor<6x!Zp1>
// 33554431 = 2 ** 25 - 1
!Zp2 = !mod_arith.int<33554431 : i26>
!Zp2v = tensor<6x!Zp2>

func.func public @test_lower_reduce_1() -> memref<6xi32> {
  // reduce intends the input to be signed
  // 67108862 = 2 ** 26 - 2, equivalent to -2 as input
  %x = arith.constant dense<[29498763, 42, 67108862, 7681, -1, 7680]> : tensor<6xi26>
  %e1 = mod_arith.encapsulate %x : tensor<6xi26> -> !Zp1v
  %m1 = mod_arith.reduce %e1 : !Zp1v
  %1 = mod_arith.extract %m1 : !Zp1v -> tensor<6xi26>

  %2 = arith.extui %1 : tensor<6xi26> to tensor<6xi32>
  %3 = bufferization.to_buffer %2 : tensor<6xi32> to memref<6xi32>
  return %3 : memref<6xi32>
}

func.func public @test_lower_reduce_2() -> memref<6xi32> {
  // 67108862 = 2 ** 26 - 2, equivalent to -2 as input
  %y = arith.constant dense<[29498763, 42, 67108862, 67108863, -1, 7680]> : tensor<6xi26>
  // 33554431 = 2 ** 25 - 1
  %e4 = mod_arith.encapsulate %y : tensor<6xi26> -> !Zp2v
  %m4 = mod_arith.reduce %e4 : !Zp2v
  %4 = mod_arith.extract %m4 : !Zp2v -> tensor<6xi26>

  %5 = arith.extui %4 : tensor<6xi26> to tensor<6xi32>
  %6 = bufferization.to_buffer %5 : tensor<6xi32> to memref<6xi32>
  return %6 : memref<6xi32>
}
