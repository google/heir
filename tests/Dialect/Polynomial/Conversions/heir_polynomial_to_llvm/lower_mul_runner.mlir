// RUN: heir-opt %s --heir-polynomial-to-llvm \
// RUN:   | mlir-cpu-runner -e test_poly_mul -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_runner_utils" > %t
// RUN: FileCheck %s --check-prefix=CHECK_TEST_POLY_MUL < %t

#ideal = #polynomial.int_polynomial<1 + x**12>
#ring = #polynomial.ring<coefficientType = i32, coefficientModulus = 4294967296 : i64, polynomialModulus=#ideal>
!poly_ty = !polynomial.polynomial<ring=#ring>

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

func.func @test_poly_mul() {
  // 1 - x^9 + x^10 + x^11
  %const0 = arith.constant 0 : index
  %0 = polynomial.constant int<1 + x**10> : !poly_ty
  %1 = polynomial.constant int<1 + x**11> : !poly_ty
  %2 = polynomial.mul %0, %1 : !poly_ty

  %3 = polynomial.to_tensor %2 : !poly_ty -> tensor<12xi32>
  %4 = bufferization.to_memref %3 : memref<12xi32>
  %U = memref.cast %4 : memref<12xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}
// CHECK_TEST_POLY_MUL: [1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 1]
