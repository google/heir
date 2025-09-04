!Zp = !mod_arith.int<3097973 : i26>
!RNS = !rns.rns<!mod_arith.int<829 : i11>, !mod_arith.int<101 : i11>, !mod_arith.int<37 : i11>>

func.func public @test_lower_mod_switch_interpolate() -> i26 {
  %x = arith.constant dense<[798, 94, 23]> : tensor<3xi11>

  %ex = mod_arith.encapsulate %x : tensor<3xi11> -> !RNS
  %m1 = mod_arith.mod_switch %ex : !RNS to !Zp
  %1 = mod_arith.extract %m1 : !Zp -> i26
  return %1 : i26
}
