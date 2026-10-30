// RUN: heir-opt --secretize --wrap-generic --canonicalize --cse --heco-simd-vectorizer %s | FileCheck %s

// The 64x64 version of roberts_cross_16x16.
// CHECK: @roberts_cross
// CHECK-NOT: affine.for
// CHECK-DAG: %[[c4095:.*]] = arith.constant 4095 : index
// CHECK-DAG: %[[c4032:.*]] = arith.constant 4032 : index
// CHECK-DAG: %[[c4031:.*]] = arith.constant 4031 : index
// CHECK: tensor_ext.rotate %{{.*}}, %[[c4031]]
// CHECK: tensor_ext.rotate %{{.*}}, %[[c4032]]
// CHECK: tensor_ext.rotate %{{.*}}, %[[c4095]]
// CHECK-NOT: tensor_ext.rotate

module{
  func.func @roberts_cross(%img: tensor<4096xi16>) -> tensor<4096xi16> {
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c-1 =  arith.constant -1 : index

    %r = affine.for %x = 0 to 64 iter_args(%imgx = %img) -> tensor<4096xi16> {
      %1 = affine.for %y = 0 to 64 iter_args(%imgy = %imgx) -> tensor<4096xi16> {

        %4 = arith.addi %x, %c-1 : index
        %5 = arith.muli %4, %c64 : index
        %6 = arith.addi %y, %c-1 : index
        %7 = arith.addi %5, %6 : index
        %8 = arith.remui %7, %c4096 : index
        %9 = tensor.extract %img[%8] : tensor<4096xi16>

        %10 = arith.muli %x, %c64 : index
        %11 = arith.addi %10, %y : index
        %12 = arith.remui %11, %c4096 : index
        %13 = tensor.extract %img[%12] : tensor<4096xi16>

        %14 = arith.subi %9, %13 : i16

        %15 = arith.addi %x, %c-1 : index
        %64 = arith.muli %15, %c64 : index
        %18 = arith.addi %64, %y : index
        %19 = arith.remui %18, %c4096 : index
        %20 = tensor.extract %img[%19] : tensor<4096xi16>

        %21 = arith.muli %x, %c64 : index
        %22 = arith.addi %y, %c-1 : index
        %23 = arith.addi %21, %22 : index
        %24 = arith.remui %23, %c4096 : index
        %25 = tensor.extract %img[%24] : tensor<4096xi16>

        %26 = arith.subi %20, %25 : i16

        %27 = arith.muli %14, %14 :  i16
        %28 = arith.muli %26, %26 :  i16

        %29 = arith.addi %27, %28 : i16

        %30 = tensor.insert %29 into %imgy[%12] : tensor<4096xi16>
        affine.yield %30: tensor<4096xi16>
      }
      affine.yield %1 : tensor<4096xi16>
    }
    return %r : tensor<4096xi16>
  }
}
