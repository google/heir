!Zp = !mod_arith.int<7681 : i26>
!Zpv = tensor<4x!Zp>

func.func public @test_lower_mul() -> tensor<4xi26> {
  // 67108862 is -2
  %x = arith.constant dense<[29498763, 42, 67108862, 7681]> : tensor<4xi26>
  // 36789492 is -30319372, 67108863 is -1
  %y = arith.constant dense<[36789492, 7234, 67108863, 7681]> : tensor<4xi26>
  %ex = mod_arith.encapsulate %x : tensor<4xi26> -> !Zpv
  %ey = mod_arith.encapsulate %y : tensor<4xi26> -> !Zpv
  %mx = mod_arith.reduce %ex : !Zpv
  %my = mod_arith.reduce %ey : !Zpv
  %m1 = mod_arith.mul %mx, %my : !Zpv
  %1 = mod_arith.extract %m1 : !Zpv -> tensor<4xi26>
  return %1 : tensor<4xi26>
}
