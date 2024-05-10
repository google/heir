// Ported from https://github.com/MarbleHE/HECO/blob/3e13744233ab0c09030a41ef98b4e061b6fa2eac/evaluation/benchmark/heco_input/robertscross_64x64.mlir

// RUN: heir-opt --secretize=entry-function=roberts_cross --wrap-generic --canonicalize --cse \
// RUN:   --heir-simd-vectorizer %s | FileCheck %s

module{
  // CHECK-LABEL: @roberts_cross
  // CHECK-SAME: (%[[arg0:.*]]: !secret.secret<tensor<4096xi16>>) -> !secret.secret<tensor<4096xi16>> {
  // CHECK-NEXT: %[[cMinusOne:.*]] = arith.constant 4095 : index
  // CHECK-NEXT: %[[cMinusRow:.*]] = arith.constant 4032 : index
  // CHECK-NEXT: %[[cMinusRowMinusOne:.*]] = arith.constant 4031 : index
  // CHECK-NEXT: secret.generic ins(%[[arg0]] : !secret.secret<tensor<4096xi16>>) {
  // CHECK-NEXT:  ^bb0(%[[arg1:.*]]: tensor<4096xi16>):
  // CHECK-NEXT:    %[[v1:.*]] = tensor_ext.rotate %[[arg1]], %[[cMinusRowMinusOne]]
  // CHECK-NEXT:    %[[v2:.*]] = arith.subi %[[v1]], %[[arg1]]
  // CHECK-NEXT:    %[[v3:.*]] = tensor_ext.rotate %[[arg1]], %[[cMinusRow]]
  // CHECK-NEXT:    %[[v4:.*]] = tensor_ext.rotate %[[arg1]], %[[cMinusOne]]
  // CHECK-NEXT:    %[[v5:.*]] = arith.subi %[[v3]], %[[v4]]
  // CHECK-DAG:     %[[v6:.*]] = arith.muli %[[v5]], %[[v5]]
  // CHECK-DAG:     %[[v7:.*]] = arith.muli %[[v2]], %[[v2]]
  // CHECK-NEXT:    %[[v8:.*]] = arith.addi %[[v7]], %[[v6]]
  func.func @roberts_cross(%img: tensor<4096xi16>) -> tensor<4096xi16> {
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c-1 =  arith.constant -1 : index

    // Each point p = img[x][y], where x is row and y is column, in the new image will equal:
    // (img[x-1][y-1] - img[x][y])^2 + (img[x-1][y] - img[x][y-1])^2
    %r = affine.for %x = 0 to 64 iter_args(%imgx = %img) -> tensor<4096xi16> {
      %1 = affine.for %y = 0 to 64 iter_args(%imgy = %imgx) -> tensor<4096xi16> {

        // fetch img[x-1][y-1]
        %4 = arith.addi %x, %c-1 : index
        %5 = arith.muli %4, %c64 : index
        %6 = arith.addi %y, %c-1 : index
        %7 = arith.addi %5, %6 : index
        %8 = arith.remui %7, %c4096 : index
        %9 = tensor.extract %img[%8] : tensor<4096xi16>

        // fetch img[x][y]
        %10 = arith.muli %x, %c64 : index
        %11 = arith.addi %10, %y : index
        %12 = arith.remui %11, %c4096 : index
        %13 = tensor.extract %img[%12] : tensor<4096xi16>

        // subtract those two
        %14 = arith.subi %9, %13 : i16

        // fetch img[x-1][y]
        %15 = arith.addi %x, %c-1 : index
        %16 = arith.muli %15, %c64 : index
        %18 = arith.addi %16, %y : index
        %19 = arith.remui %18, %c4096 : index
        %20 = tensor.extract %img[%19] : tensor<4096xi16>

        // fetch img[x][y-1]
        %21 = arith.muli %x, %c64 : index
        %22 = arith.addi %y, %c-1 : index
        %23 = arith.addi %21, %22 : index
        %24 = arith.remui %23, %c4096 : index
        %25 = tensor.extract %img[%24] : tensor<4096xi16>

        // subtract those two
        %26 = arith.subi %20, %25 : i16

        // square each difference
        %27 = arith.muli %14, %14 :  i16
        %28 = arith.muli %26, %26 :  i16

        // add the squares
        %29 = arith.addi %27, %28 : i16

        // save to result[x][y]
        %30 = tensor.insert %29 into %imgy[%12] : tensor<4096xi16>
        affine.yield %30: tensor<4096xi16>
      }
      affine.yield %1 : tensor<4096xi16>
    }
    return %r : tensor<4096xi16>
  }
}
