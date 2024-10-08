// RUN: heir-opt %s --mod-arith-to-arith --heir-polynomial-to-llvm \
// RUN:   | mlir-cpu-runner -e test_lower_sub -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_runner_utils" > %t
// RUN: FileCheck %s --check-prefix=CHECK_TEST_SUB < %t

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

func.func @test_lower_sub() {
  %x = arith.constant dense<[29498763, 42, 67108862, 7681]> : tensor<4xi26>
  %y = arith.constant dense<[36789492, 7234, 67108863, 7681]> : tensor<4xi26>
  %1 = mod_arith.sub %x, %y { modulus = 7681 } : tensor<4xi26>

  %2 = arith.extui %1 : tensor<4xi26> to tensor<4xi32>
  %3 = bufferization.to_memref %2 : memref<4xi32>
  %U = memref.cast %3 : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}

// CHECK_TEST_SUB: [6188, 489, 7680, 0]
