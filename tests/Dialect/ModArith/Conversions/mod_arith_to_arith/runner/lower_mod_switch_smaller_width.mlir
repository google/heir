!Zp = !mod_arith.int<3097973 : i26>
!Zp_smaller_width = !mod_arith.int<257 : i10>

func.func public @lower_mod_switch_smaller_width() -> i10 {
  %x = arith.constant 1214522 : i26
  %ex = mod_arith.encapsulate %x : i26 -> !Zp
  %mx = mod_arith.reduce %ex : !Zp
  %m1 = mod_arith.mod_switch %mx : !Zp to !Zp_smaller_width
  %1 = mod_arith.extract %m1 : !Zp_smaller_width -> i10
  return %1 : i10
}
