// RUN: heir-opt %s --mod-arith-to-arith --heir-polynomial-to-llvm \
// RUN:   | mlir-runner -e lower_mod_switch_same_width -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_runner_utils" > %t
// RUN: FileCheck %s --check-prefix=CHECK_TEST_MOD_SWITCH_SAME_WIDTH < %t

llvm.func @printI64(i64)
llvm.func @printNewline()

!Zp = !mod_arith.int<3097973 : i26>
!Zp_same_width = !mod_arith.int<33181787 : i26>

func.func @lower_mod_switch_same_width() {
  %x = arith.constant 2241752 : i26
  %ex = mod_arith.encapsulate %x : i26 -> !Zp
  %mx = mod_arith.reduce %ex : !Zp
  %m1 = mod_arith.mod_switch %mx : !Zp to !Zp_same_width
  %1 = mod_arith.extract %m1 : !Zp_same_width -> i26

  %2 = arith.extui %1 : i26 to i64
  llvm.call @printI64(%2) : (i64) -> ()
  llvm.call @printNewline() : () -> ()
  return
}

// CHECK_TEST_MOD_SWITCH_SAME_WIDTH: 32325566
