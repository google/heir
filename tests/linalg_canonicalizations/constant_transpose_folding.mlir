// RUN: heir-opt --linalg-canonicalizations %s | FileCheck %s

func.func @transpose_constant_fold(%init1: tensor<3x4x2xi16>,
                                   %init2: tensor<2x3xf16>) -> (tensor<3x4x2xi16>, tensor<2x3xf16>) {
  //         CHECK-LABEL: @transpose_constant_fold
  //               CHECK: %[[FOLDED_CONSTANT1:[a-zA-Z0-9_]+]] = arith.constant dense
  // CHECK-SAME{LITERAL}: <[[[1, 13], [2, 14], [3, 15], [4, 16]], [[5, 17], [6, 18], [7, 19], [8, 20]], [[9, 21], [10, 22], [11, 23], [12, 24]]]> : tensor<3x4x2xi16>
  //           CHECK-NOT: linalg.transpose
  //               CHECK: %[[FOLDED_CONSTANT2:[a-zA-Z0-9_]+]] = arith.constant dense
  // CHECK-SAME{LITERAL}:   <[[
  //          CHECK-SAME:   1.{{0+}}e+00, 3.{{0+}}e+00, 5.{{0+}}e+00], [2.{{0+}}e+00, 4.{{0+}}e+00, 6.{{0+}}e+00
  // CHECK-SAME{LITERAL}:   ]]> : tensor<2x3xf16>
  //           CHECK-NOT: linalg.transpose
  //               CHECK: return %[[FOLDED_CONSTANT1]], %[[FOLDED_CONSTANT2]] : tensor<3x4x2xi16>, tensor<2x3xf16>
  %constant1 = arith.constant dense<[
      [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
      [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]>
      : tensor<2x3x4xi16>
  %transpose1 = linalg.transpose
      ins(%constant1:tensor<2x3x4xi16>)
      outs(%init1:tensor<3x4x2xi16>)
      permutation = [1, 2, 0]

  %constant2 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]> : tensor<3x2xf16>
  %transpose2 = linalg.transpose
      ins(%constant2:tensor<3x2xf16>)
      outs(%init2:tensor<2x3xf16>)
      permutation = [1, 0]
  func.return %transpose1, %transpose2 : tensor<3x4x2xi16>, tensor<2x3xf16>
}
