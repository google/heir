// RUN: heir-opt %s --heir-polynomial-to-llvm \
// RUN:   | mlir-cpu-runner -e test_poly_ntt -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_runner_utils" > %t
// RUN: FileCheck %s --check-prefix=CHECK_TEST_POLY_NTT < %t

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

#cycl = #polynomial.polynomial<1 + x**4>
#ring = #polynomial.ring<cmod=7681, ideal=#cycl, root=1925>
!poly_ty = !polynomial.polynomial<#ring>

func.func @test_poly_ntt() {
  %coeffs = arith.constant dense<[1,2,3,4]> : tensor<4xi13>
  %poly = polynomial.from_tensor %coeffs : tensor<4xi13> -> !poly_ty
  %0 = polynomial.ntt %poly : !poly_ty -> tensor<4xi13, #ring>

  %1 = tensor.cast %0 : tensor<4xi13, #ring> to tensor<4xi13>
  %2 = arith.extui %1 : tensor<4xi13> to tensor<4xi32>
  %3 = bufferization.to_memref %2 : memref<4xi32>
  %U = memref.cast %3 : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}
// CHECK_TEST_POLY_NTT: [1467, 2807, 3471, 7621]
