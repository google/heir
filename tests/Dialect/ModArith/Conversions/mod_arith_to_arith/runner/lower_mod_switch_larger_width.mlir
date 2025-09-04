!Zp = !mod_arith.int<3097973 : i26>
!Zp_larger_width = !mod_arith.int<65537 : i32>

func.func public @lower_mod_switch_larger_width() -> i32 {
  %x = arith.constant 2241752 : i26
  %ex = mod_arith.encapsulate %x : i26 -> !Zp
  %mx = mod_arith.reduce %ex : !Zp
  %m1 = mod_arith.mod_switch %mx : !Zp to !Zp_larger_width
  %1 = mod_arith.extract %m1 : !Zp_larger_width -> i32
  return %1 : i32
}
