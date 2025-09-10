!Zp = !mod_arith.int<3097973 : i26>
!RNS = !rns.rns<!mod_arith.int<829 : i11>, !mod_arith.int<101 : i11>, !mod_arith.int<37 : i11>>

func.func public @test_lower_mod_switch_decompose() -> tensor<3xi11> {
  // 57543298 is -9565566
  %x = arith.constant 57543298 : i26
  %ex = mod_arith.encapsulate %x : i26 -> !Zp
  %mx = mod_arith.reduce %ex : !Zp
  %m1 = mod_arith.mod_switch %mx : !Zp to !RNS
  %1 = mod_arith.extract %m1 : !RNS -> tensor<3xi11>
  return %1 : tensor<3xi11>
}
