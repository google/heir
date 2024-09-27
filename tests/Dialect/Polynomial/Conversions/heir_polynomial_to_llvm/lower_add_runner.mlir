// RUN: heir-opt %s --heir-polynomial-to-llvm \
// RUN:   | mlir-cpu-runner -e test_poly_add -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_runner_utils" > %t
// RUN: FileCheck %s --check-prefix=CHECK_TEST_POLY_ADD < %t

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

func.func @test_poly_add() {
  %const0 = arith.constant 0 : index
  %0 = polynomial.constant int<1 + x**10> : !poly
  %1 = polynomial.constant int<1 + x**11> : !poly
  %2 = polynomial.add %0, %1 : !poly

  %3 = polynomial.to_tensor %2 : !poly -> tensor<12xi32>
  %4 = bufferization.to_memref %3 : memref<12xi32>
  %U = memref.cast %4 : memref<12xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  // CHECK_TEST_POLY_ADD: rank = 1 offset = 0 sizes = [12] strides = [1] data =
  // CHECK_TEST_POLY_ADD: [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]

  %5 = polynomial.constant int<1 + -1 x + x**10> : !poly_no
  %6 = polynomial.constant int<1 + x**11> : !poly_no
  %7 = polynomial.add %5, %6 : !poly_no

  %8 = polynomial.to_tensor %7 : !poly_no -> tensor<12xi32>
  %9 = bufferization.to_memref %8 : memref<12xi32>
  %V = memref.cast %9 : memref<12xi32> to memref<*xi32>
  func.call @printMemrefI32(%V) : (memref<*xi32>) -> ()
  // CHECK_TEST_POLY_ADD: rank = 1 offset = 0 sizes = [12] strides = [1] data =
  // CHECK_TEST_POLY_ADD: [2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]

  %10 = polynomial.constant int<1 + -1 x + 16 x**2+ x**10> : !poly_small
  %11 = polynomial.constant int<1 + 2 x**2 + x**11> : !poly_small
  %12 = polynomial.add %10, %11 : !poly_small

  %13 = polynomial.to_tensor %12 : !poly_small -> tensor<12xi32>
  %14 = bufferization.to_memref %13 : memref<12xi32>
  %W = memref.cast %14 : memref<12xi32> to memref<*xi32>
  func.call @printMemrefI32(%W) : (memref<*xi32>) -> ()
  // CHECK_TEST_POLY_ADD: rank = 1 offset = 0 sizes = [12] strides = [1] data =
  // CHECK_TEST_POLY_ADD: [2, 16, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]

  // unfortunately, 4294967290 is a signed number, thus equal to -6
  %15 = polynomial.constant int<1 + -1 x + 4294967290 x**2+ x**10> : !poly_prime
  %16 = polynomial.constant int<1 + x**2 + x**11> : !poly_prime
  %17 = polynomial.add %15, %16 : !poly_prime

  %18 = polynomial.to_tensor %17 : !poly_prime -> tensor<12xi32>
  %19 = bufferization.to_memref %18 : memref<12xi32>
  %Z = memref.cast %19 : memref<12xi32> to memref<*xi32>
  func.call @printMemrefI32(%Z) : (memref<*xi32>) -> ()
  // CHECK_TEST_POLY_ADD: rank = 1 offset = 0 sizes = [12] strides = [1] data =
  // should be [2, 4294967291 - 1, 4294967291 - 5, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  // however, i32 is printed with sign and 4294967291 equals -5
  // very counterintuitive result!
  // CHECK_TEST_POLY_ADD: [2, {{4294967290|-6}}, {{4294967286|-10}}, 0, 0, 0, 0, 0, 0, 0, 1, 1]

  return
}