// RUN: heir-opt %s --heir-polynomial-to-llvm \
// RUN:   | mlir-cpu-runner -e test_poly_ntt -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_runner_utils" > %t
// RUN: FileCheck %s --check-prefix=CHECK_TEST_POLY_NTT < %t

// This follows from example 3.10 (Satriawan et al.) here:
// https://doi.org/10.1109/ACCESS.2023.3294446

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

#cycl = #_polynomial.polynomial<1 + x**4>
#ring = #_polynomial.ring<cmod=7681, ideal=#cycl, root=1925>
!poly_ty = !_polynomial.polynomial<#ring>

func.func @test_poly_ntt() {
  %coeffs = arith.constant dense<[1467,2807,3471,7621]> : tensor<4xi13>
  %ntt_coeffs = tensor.cast %coeffs : tensor<4xi13> to tensor<4xi13, #ring>
  %0 = _polynomial.intt %ntt_coeffs : tensor<4xi13, #ring> -> !poly_ty

  %1 = _polynomial.to_tensor %0 : !poly_ty -> tensor<4xi13>
  %2 = arith.extui %1 : tensor<4xi13> to tensor<4xi32>
  %3 = bufferization.to_memref %2 : memref<4xi32>
  %U = memref.cast %3 : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}
// CHECK_TEST_POLY_NTT: [1, 2, 3, 4]
