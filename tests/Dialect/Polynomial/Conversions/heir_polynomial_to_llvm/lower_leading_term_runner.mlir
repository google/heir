// RUN: heir-opt %s --heir-polynomial-to-llvm \
// RUN:   | mlir-cpu-runner -e test_poly_leading_term -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_runner_utils" > %t
// RUN: FileCheck %s --check-prefix=CHECK_TEST_POLY_LEADING_TERM < %t

#ideal = #polynomial.int_polynomial<1 + x**12>
#ring = #polynomial.ring<coefficientType = i32, coefficientModulus = 4294967296 : i64, polynomialModulus=#ideal>
!poly_ty = !polynomial.polynomial<ring=#ring>

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

func.func @test_poly_leading_term() {
  %const0 = arith.constant 0 : index
  %0 = polynomial.constant int<1 + 2x**10> : !poly_ty
  %2, %1 = polynomial.leading_term %0 : !poly_ty -> (index, i32)

  %3 = memref.alloca() : memref<2xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  memref.store %1, %3[%c0] : memref<2xi32>
  %4 = arith.index_cast %2 : index to i32
  memref.store %4, %3[%c1] : memref<2xi32>

  %U = memref.cast %3 : memref<2xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}
// CHECK_TEST_POLY_LEADING_TERM: rank = 1 offset = 0 sizes = [2] strides = [1] data =
// CHECK_TEST_POLY_LEADING_TERM: [2, 10]
