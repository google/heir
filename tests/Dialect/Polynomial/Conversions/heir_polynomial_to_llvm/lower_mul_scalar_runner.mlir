// RUN: heir-opt %s --heir-polynomial-to-llvm \
// RUN:   | mlir-cpu-runner -e test_poly_mul_scalar -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_runner_utils" > %t
// RUN: FileCheck %s --check-prefix=CHECK_TEST_POLY_MUL_SCALAR < %t

#ideal = #polynomial.int_polynomial<1 + x**12>
#ring = #polynomial.ring<coefficientType = i32, coefficientModulus = 4294967296 : i64, polynomialModulus=#ideal>
#ring_no = #polynomial.ring<coefficientType = i32, polynomialModulus=#ideal>
#ring_small = #polynomial.ring<coefficientType = i32, coefficientModulus = 17 : i64, polynomialModulus=#ideal>
#ring_prime = #polynomial.ring<coefficientType = i32, coefficientModulus = 4294967291 : i64, polynomialModulus=#ideal>

!poly = !polynomial.polynomial<ring=#ring>
!poly_no = !polynomial.polynomial<ring=#ring_no>
!poly_small = !polynomial.polynomial<ring=#ring_small>
!poly_prime = !polynomial.polynomial<ring=#ring_prime>

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

func.func @test_poly_mul_scalar() {
  %const0 = arith.constant 0 : index
  %0 = polynomial.constant int<1 + -1 x + x**11> : !poly
  %1 = arith.constant -2 : i32
  %2 = polynomial.mul_scalar %0, %1 : !poly, i32

  %3 = polynomial.to_tensor %2 : !poly -> tensor<12xi32>
  %4 = bufferization.to_memref %3 : memref<12xi32>
  %U = memref.cast %4 : memref<12xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  // CHECK_TEST_POLY_MUL_SCALAR: rank = 1 offset = 0 sizes = [12] strides = [1] data =
  // CHECK_TEST_POLY_MUL_SCALAR: [-2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2]

  %5 = polynomial.constant int<1 + -1 x + x**10> : !poly_no
  %6 = arith.constant -2 : i32
  %7 = polynomial.mul_scalar %5, %6 : !poly_no, i32

  %8 = polynomial.to_tensor %7 : !poly_no -> tensor<12xi32>
  %9 = bufferization.to_memref %8 : memref<12xi32>
  %V = memref.cast %9 : memref<12xi32> to memref<*xi32>
  func.call @printMemrefI32(%V) : (memref<*xi32>) -> ()
  // CHECK_TEST_POLY_MUL_SCALAR: rank = 1 offset = 0 sizes = [12] strides = [1] data =
  // CHECK_TEST_POLY_MUL_SCALAR: [-2, 2, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0]

  %10 = polynomial.constant int<1 + -1 x + 16 x**2+ x**10> : !poly_small
  %11 = arith.constant -2 : i32
  %12 = polynomial.mul_scalar %10, %11 : !poly_small, i32

  %13 = polynomial.to_tensor %12 : !poly_small -> tensor<12xi32>
  %14 = bufferization.to_memref %13 : memref<12xi32>
  %W = memref.cast %14 : memref<12xi32> to memref<*xi32>
  func.call @printMemrefI32(%W) : (memref<*xi32>) -> ()
  // CHECK_TEST_POLY_MUL_SCALAR: rank = 1 offset = 0 sizes = [12] strides = [1] data =
  // CHECK_TEST_POLY_MUL_SCALAR: [15, 2, 2, 0, 0, 0, 0, 0, 0, 0, 15, 0]

  // unfortunately, 4294967290 is a signed number, thus equal to -6
  %15 = polynomial.constant int<1 + -1 x + 4294967290 x**2+ x**10> : !poly_prime
  %16 = arith.constant -2 : i32
  %17 = polynomial.mul_scalar %15, %16 : !poly_prime, i32

  %18 = polynomial.to_tensor %17 : !poly_prime -> tensor<12xi32>
  %19 = bufferization.to_memref %18 : memref<12xi32>
  %Z = memref.cast %19 : memref<12xi32> to memref<*xi32>
  func.call @printMemrefI32(%Z) : (memref<*xi32>) -> ()
  // CHECK_TEST_POLY_MUL_SCALAR: rank = 1 offset = 0 sizes = [12] strides = [1] data =
  // should be [4294967291 - 2, 2, 12, 0, 0, 0, 0, 0, 0, 0, 4294967291 - 2, 0]
  // however, i32 is printed with sign and 4294967291 equals -5
  // very counterintuitive result!
  // CHECK_TEST_POLY_MUL_SCALAR: [{{4294967289|-7}}, 2, 12, 0, 0, 0, 0, 0, 0, 0, -7, 0]

  return
}
