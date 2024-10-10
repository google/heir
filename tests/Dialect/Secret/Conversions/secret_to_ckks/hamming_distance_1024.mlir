// RUN: heir-opt --secret-distribute-generic --secret-to-ckks %s | FileCheck %s

// CHECK-LABEL: @hamming
// CHECK: ckks.sub
// CHECK-NEXT: ckks.mul
// CHECK-NEXT: ckks.relinearize
// CHECK-COUNT-10: ckks.rotate
// CHECK: ckks.extract
// CHECK-NEXT: return

func.func @hamming(%arg0: !secret.secret<tensor<1024xi16>>, %arg1: !secret.secret<tensor<1024xi16>>) -> !secret.secret<i16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c512 = arith.constant 512 : index
  %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<tensor<1024xi16>>, !secret.secret<tensor<1024xi16>>) {
  ^bb0(%arg2: tensor<1024xi16>, %arg3: tensor<1024xi16>):
    %1 = arith.subi %arg2, %arg3 : tensor<1024xi16>
    %2 = arith.muli %1, %1 : tensor<1024xi16>
    %3 = tensor_ext.rotate %2, %c512 : tensor<1024xi16>, index
    %4 = arith.addi %2, %3 : tensor<1024xi16>
    %5 = tensor_ext.rotate %4, %c256 : tensor<1024xi16>, index
    %6 = arith.addi %4, %5 : tensor<1024xi16>
    %7 = tensor_ext.rotate %6, %c128 : tensor<1024xi16>, index
    %8 = arith.addi %6, %7 : tensor<1024xi16>
    %9 = tensor_ext.rotate %8, %c64 : tensor<1024xi16>, index
    %10 = arith.addi %8, %9 : tensor<1024xi16>
    %11 = tensor_ext.rotate %10, %c32 : tensor<1024xi16>, index
    %12 = arith.addi %10, %11 : tensor<1024xi16>
    %13 = tensor_ext.rotate %12, %c16 : tensor<1024xi16>, index
    %14 = arith.addi %12, %13 : tensor<1024xi16>
    %15 = tensor_ext.rotate %14, %c8 : tensor<1024xi16>, index
    %16 = arith.addi %14, %15 : tensor<1024xi16>
    %17 = tensor_ext.rotate %16, %c4 : tensor<1024xi16>, index
    %18 = arith.addi %16, %17 : tensor<1024xi16>
    %19 = tensor_ext.rotate %18, %c2 : tensor<1024xi16>, index
    %20 = arith.addi %18, %19 : tensor<1024xi16>
    %21 = tensor_ext.rotate %20, %c1 : tensor<1024xi16>, index
    %22 = arith.addi %20, %21 : tensor<1024xi16>
    %extracted = tensor.extract %22[%c0] : tensor<1024xi16>
    secret.yield %extracted : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}
