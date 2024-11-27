// RUN: heir-opt %s --mod-arith-to-arith --heir-polynomial-to-llvm \
// RUN:   | mlir-cpu-runner -e test_lower_reduce -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_runner_utils" > %t
// RUN: FileCheck %s --check-prefix=CHECK_TEST_REDUCE < %t

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

!Zp1 = !mod_arith.int<7681 : i26>
!Zp1v = tensor<6x!Zp1>
// 33554431 = 2 ** 25 - 1
!Zp2 = !mod_arith.int<33554431 : i26>
!Zp2v = tensor<6x!Zp2>

func.func @test_lower_reduce() {
  // reduce intends the input to be signed
  // 67108862 = 2 ** 26 - 2, equivalent to -2 as input
  %x = arith.constant dense<[29498763, 42, 67108862, 7681, -1, 7680]> : tensor<6xi26>
  %e1 = mod_arith.encapsulate %x : tensor<6xi26> -> !Zp1v
  %m1 = mod_arith.reduce %e1 : !Zp1v
  %1 = mod_arith.extract %m1 : !Zp1v -> tensor<6xi26>
  // CHECK_TEST_REDUCE: [3723, 42, 7679, 0, 7680, 7680]

  %2 = arith.extui %1 : tensor<6xi26> to tensor<6xi32>
  %3 = bufferization.to_memref %2 : tensor<6xi32> to memref<6xi32>
  %U = memref.cast %3 : memref<6xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()


  // 67108862 = 2 ** 26 - 2, equivalent to -2 as input
  %y = arith.constant dense<[29498763, 42, 67108862, 67108863, -1, 7680]> : tensor<6xi26>
  // 33554431 = 2 ** 25 - 1
  %e4 = mod_arith.encapsulate %y : tensor<6xi26> -> !Zp2v
  %m4 = mod_arith.reduce %e4 : !Zp2v
  %4 = mod_arith.extract %m4 : !Zp2v -> tensor<6xi26>
  // CHECK_TEST_REDUCE: [29498763, 42, 33554429, 33554430, 33554430, 7680]

  %5 = arith.extui %4 : tensor<6xi26> to tensor<6xi32>
  %6 = bufferization.to_memref %5 : tensor<6xi32> to memref<6xi32>
  %V = memref.cast %6 : memref<6xi32> to memref<*xi32>
  func.call @printMemrefI32(%V) : (memref<*xi32>) -> ()
  return
}
