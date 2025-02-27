// RUN: heir-opt %s --mlir-to-secret-arithmetic | FileCheck %s --check-prefix=CHECK_MLIR_TO_SECRET_ARITHMETIC

// RUN: heir-opt %s --mlir-to-secret-arithmetic --tensor-ext-to-tensor --secret-forget-secrets --heir-polynomial-to-llvm \
// RUN:   | mlir-runner -e test_dot_product -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_runner_utils" > %t
// RUN: FileCheck %s --check-prefix=CHECK_TEST_DOT_PRODUCT < %t

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

// CHECK_MLIR_TO_SECRET_ARITHMETIC: func.func @dot_product(%arg0: !secret.secret<tensor<8xi16>>
// CHECK_MLIR_TO_SECRET_ARITHMETIC-NOT: affine.for

func.func @dot_product(%arg0: tensor<8xi16> {secret.secret}, %arg1: tensor<8xi16> {secret.secret}) -> i16 {
  %c0 = arith.constant 0 : index
  %c0_si16 = arith.constant 0 : i16
  %0 = affine.for %arg2 = 0 to 8 iter_args(%iter = %c0_si16) -> (i16) {
    %1 = tensor.extract %arg0[%arg2] : tensor<8xi16>
    %2 = tensor.extract %arg1[%arg2] : tensor<8xi16>
    %3 = arith.muli %1, %2 : i16
    %4 = arith.addi %iter, %3 : i16
    affine.yield %4 : i16
  }
  return %0 : i16
}

func.func @test_dot_product() {
  %arg0 = arith.constant dense<[1, 2, 3, 4, 5, 6, 7, 8]> : tensor<8xi16>
  %arg1 = arith.constant dense<[2, 3, 4, 5, 6, 7, 8, 9]> : tensor<8xi16>

  %res = func.call @dot_product(%arg0, %arg1) : (tensor<8xi16>, tensor<8xi16>) -> i16
  %res_i32 = arith.extui %res : i16 to i32
  %res_tensor = tensor.from_elements %res_i32 : tensor<1xi32>
  %memref = bufferization.to_memref %res_tensor : tensor<1xi32> to memref<1xi32>
  %ptr = memref.cast %memref : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%ptr) : (memref<*xi32>) -> ()
  return
}
// CHECK_TEST_DOT_PRODUCT: rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_TEST_DOT_PRODUCT: [240]
