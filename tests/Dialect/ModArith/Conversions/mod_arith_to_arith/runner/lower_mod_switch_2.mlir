// RUN: heir-opt %s --mod-arith-to-arith --heir-polynomial-to-llvm \
// RUN:   | mlir-runner -e test_lower_mod_switch_2 -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_runner_utils" > %t
// RUN: FileCheck %s --check-prefix=CHECK_TEST_MOD_SWITCH_2 < %t

llvm.func @printI64(i64)
llvm.func @printNewline()

!RNS = !rns.rns<!mod_arith.int<829 : i11>, !mod_arith.int<101 : i11>, !mod_arith.int<37 : i11>>
!Zp = !mod_arith.int<3097973 : i26>

func.func @test_lower_mod_switch_2() {
  // 57543298 is -9565566
  %x = arith.constant dense<[690, 48, 16]> : tensor<3xi11>
  %ex = mod_arith.encapsulate %x : tensor<3xi11> -> !RNS
  %m1 = mod_arith.mod_switch %mx : !RNS to !Zp
  %1 = mod_arith.extract %m1 : !Zp -> i26

  %2 = arith.extui %1 : i26 to i64
  llvm.call @printI64(%2) : (i64) -> ()
  llvm.call @printNewline() : () -> ()
  return
}

// CHECK_TEST_MOD_SWITCH_2: 1470507
