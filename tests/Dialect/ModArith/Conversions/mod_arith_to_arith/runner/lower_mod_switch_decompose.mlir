// RUN: heir-opt %s --mod-arith-to-arith --heir-polynomial-to-llvm \
// RUN:   | mlir-runner -e test_lower_mod_switch_decompose -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_runner_utils" > %t
// RUN: FileCheck %s --check-prefix=CHECK_TEST_MOD_SWITCH_DECOMPOSE < %t

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

!Zp = !mod_arith.int<3097973 : i26>
!RNS = !rns.rns<!mod_arith.int<829 : i11>, !mod_arith.int<101 : i11>, !mod_arith.int<37 : i11>>

func.func @test_lower_mod_switch_decompose() {
  // 57543298 is -9565566
  %x = arith.constant 57543298 : i26
  %ex = mod_arith.encapsulate %x : i26 -> !Zp
  %mx = mod_arith.reduce %ex : !Zp
  %m1 = mod_arith.mod_switch %mx : !Zp to !RNS
  %1 = mod_arith.extract %m1 : !RNS -> tensor<3xi11>

  %2 = arith.extui %1 : tensor<3xi11> to tensor<3xi32>
  %3 = bufferization.to_buffer %2 : tensor<3xi32> to memref<3xi32>
  %U = memref.cast %3 : memref<3xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}

// CHECK_TEST_MOD_SWITCH_DECOMPOSE: [265, 43, 7]
