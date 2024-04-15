// RUN: heir-opt %s --heir-polynomial-to-llvm \
// RUN:   | mlir-cpu-runner -e test_poly_add -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_runner_utils" > %t
// RUN: FileCheck %s --check-prefix=CHECK_TEST_POLY_ADD < %t

#ideal = #_polynomial.polynomial<1 + x**12>
#ring = #_polynomial.ring<cmod=4294967296, ideal=#ideal>
!poly_ty = !_polynomial.polynomial<#ring>

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

func.func @test_poly_add() {
  %const0 = arith.constant 0 : index
  %0 = _polynomial.constant <1 + x**10> : !poly_ty
  %1 = _polynomial.constant <1 + x**11> : !poly_ty
  %2 = _polynomial.add(%0, %1) : !poly_ty

  %3 = _polynomial.to_tensor %2 : !poly_ty -> tensor<12xi32>
  %4 = bufferization.to_memref %3 : memref<12xi32>
  %U = memref.cast %4 : memref<12xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}
// CHECK_TEST_POLY_ADD: rank = 1 offset = 0 sizes = [12] strides = [1] data =
// CHECK_TEST_POLY_ADD: [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
