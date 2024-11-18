// RUN: heir-opt %s --mod-arith-to-arith --heir-polynomial-to-llvm \
// RUN:   | mlir-cpu-runner -e test_lower_barrett_reduce -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_runner_utils" > %t
// RUN: FileCheck %s --check-prefix=CHECK_TEST_BARRETT < %t

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

!Zp = !mod_arith.int<7681 : i32>
!Zpv = tensor<4x!Zp>

func.func @test_lower_barrett_reduce() {
  %coeffs = arith.constant dense<[29498763, 58997760, 17, 7681]> : tensor<4xi64>
  %1 = mod_arith.barrett_reduce %coeffs  : tensor<4xi64> -> !Zpv

  %2 = mod_arith.extract %1 : !Zpv -> tensor<4xi32>

  // %3 = arith.extui %2 : tensor<4xi32> to tensor<4xi32>
  %4 = bufferization.to_memref %2 : memref<4xi32>
  %U = memref.cast %4 : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}


// CHECK_TEST_BARRETT: [3723, 7680, 17, 7681]
