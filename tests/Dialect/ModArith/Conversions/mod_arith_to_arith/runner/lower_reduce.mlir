// RUN: heir-opt %s --mod-arith-to-arith --heir-polynomial-to-llvm \
// RUN:   | mlir-cpu-runner -e test_lower_reduce -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_runner_utils" > %t
// RUN: FileCheck %s --check-prefix=CHECK_TEST_REDUCE < %t

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

func.func @test_lower_reduce() {
  // reduce intends the input to be signed
  // 67108862 = 2 ** 26 - 2, equivalent to -2 as input
  %x = arith.constant dense<[29498763, 42, 67108862, 7681, -1, 7680]> : tensor<6xi26>
  %1 = mod_arith.reduce %x { modulus = 7681 } : tensor<6xi26>
  // CHECK_TEST_REDUCE: [3723, 42, 7679, 0, 7680, 7680]

  %2 = arith.extui %1 : tensor<6xi26> to tensor<6xi32>
  %3 = bufferization.to_memref %2 : memref<6xi32>
  %U = memref.cast %3 : memref<6xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()


  // 67108862 = 2 ** 26 - 2, equivalent to -2 as input
  %y = arith.constant dense<[29498763, 42, 67108862, 67108863, -1, 7680]> : tensor<6xi26>
  // 33554432 = 2 ** 25
  %4 = mod_arith.reduce %y { modulus = 33554432 } : tensor<6xi26>
  // CHECK_TEST_REDUCE: [29498763, 42, 33554430, 33554431, 33554431, 7680]

  %5 = arith.extui %4 : tensor<6xi26> to tensor<6xi32>
  %6 = bufferization.to_memref %5 : memref<6xi32>
  %V = memref.cast %6 : memref<6xi32> to memref<*xi32>
  func.call @printMemrefI32(%V) : (memref<*xi32>) -> ()

  // 33554431 = 2 ** 25 - 1
  %7 = mod_arith.reduce %y { modulus = 33554431 } : tensor<6xi26>
  // CHECK_TEST_REDUCE: [29498763, 42, 33554429, 33554430, 33554430, 7680]

  %8 = arith.extui %7 : tensor<6xi26> to tensor<6xi32>
  %9 = bufferization.to_memref %8 : memref<6xi32>
  %W = memref.cast %9 : memref<6xi32> to memref<*xi32>
  func.call @printMemrefI32(%W) : (memref<*xi32>) -> ()
  return
}
