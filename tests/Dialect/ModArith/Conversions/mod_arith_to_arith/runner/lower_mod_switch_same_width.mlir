!Zp = !mod_arith.int<3097973 : i26>
!Zp_same_width = !mod_arith.int<33181787 : i26>

func.func public @lower_mod_switch_same_width() -> i26 {
  %x = arith.constant 2241752 : i26
  %ex = mod_arith.encapsulate %x : i26 -> !Zp
  %mx = mod_arith.reduce %ex : !Zp
  %m1 = mod_arith.mod_switch %mx : !Zp to !Zp_same_width
  %1 = mod_arith.extract %m1 : !Zp_same_width -> i26
  return %1 : i26
}
