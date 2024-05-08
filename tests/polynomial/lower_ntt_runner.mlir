// RUN: heir-opt %s --heir-polynomial-to-llvm \
// RUN:   | mlir-cpu-runner -e test_poly_ntt -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_runner_utils" > %t
// RUN: FileCheck %s --check-prefix=CHECK_TEST_POLY_NTT < %t

// This follows from example 3.8 (Satriawan et al.) here:
// https://doi.org/10.1109/ACCESS.2023.3294446

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

#cycl = #polynomial.int_polynomial<1 + x**4>
#ring = #polynomial.ring<coefficientType = i32, coefficientModulus = 7681 : i32, polynomialModulus=#cycl>
#root = #polynomial.primitive_root<value=1925:i32, degree=8:i32>
!poly_ty = !polynomial.polynomial<ring=#ring>

func.func @test_poly_ntt() {
  %coeffs = arith.constant dense<[1,2,3,4]> : tensor<4xi32>
  %poly = polynomial.from_tensor %coeffs : tensor<4xi32> -> !poly_ty
  %0 = polynomial.ntt %poly {root=#root} : !poly_ty -> tensor<4xi32, #ring>

  %1 = tensor.cast %0 : tensor<4xi32, #ring> to tensor<4xi32>
  %2 = bufferization.to_memref %1 : memref<4xi32>
  %U = memref.cast %2 : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}
// CHECK_TEST_POLY_NTT: [1467, 2807, 3471, 7621]
