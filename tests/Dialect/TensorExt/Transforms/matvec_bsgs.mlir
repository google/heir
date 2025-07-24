// RUN: heir-opt --baby-step-giant-step --canonicalize %s | FileCheck %s

#layout = #tensor_ext.layout<map = (d0) -> (d0 mod 16)>
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 mod 16)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> ((d1 - d0) mod 16, (d1 - (d1 - d0) mod 16) mod 16)>
#original_type = #tensor_ext.original_type<originalType = tensor<16xi16>, layout = #layout>
module {
  // CHECK: @matvec_constant_matrix
  // CHECK-SAME: %[[ARG0:.*]]: !secret.secret<tensor<16xi16>> {tensor_ext.original_type = #original_type}
  func.func @matvec_constant_matrix(%arg0: !secret.secret<tensor<16xi16>> {tensor_ext.original_type = #original_type}) -> (!secret.secret<tensor<16xi16>> {tensor_ext.original_type = #original_type}) {
    // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[c2:.*]] = arith.constant 2 : index
    // CHECK-DAG: %[[c3:.*]] = arith.constant 3 : index
    %cst = arith.constant dense<1> : tensor<16x16xi16>
    %cst_0 = arith.constant dense<0> : tensor<16xi16>
    // CHECK: secret.generic(%[[ARG0]]: !secret.secret<tensor<16xi16>>)
    // CHECK-NEXT: ^body(%[[INPUT0:.*]]: tensor<16xi16>)
    %0 = secret.generic(%arg0: !secret.secret<tensor<16xi16>>) {
    ^body(%input0: tensor<16xi16>):
      // CHECK-DAG: %[[rot1:.*]] = tensor_ext.rotate %[[INPUT0]], %[[c1]] : tensor<16xi16>, index
      // CHECK-DAG: %[[rot2:.*]] = tensor_ext.rotate %[[INPUT0]], %[[c2]] : tensor<16xi16>, index
      // CHECK-DAG: %[[rot3:.*]] = tensor_ext.rotate %[[INPUT0]], %[[c3]] : tensor<16xi16>, index
      // There will be n - sqrt(n) plaintext rotations
      // CHECK-COUNT-12: tensor_ext.rotate
      // Then there will be sqrt(n) - 1 ciphertext rotations
      // CHECK-COUNT-3: tensor_ext.rotate
      // CHECK-NOT: tensor_ext.rotate
      // CHECK: return
      %cst_1 = arith.constant dense<0> : tensor<16xi16>
      %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins(%cst_0 : tensor<16xi16>) outs(%cst_1 : tensor<16xi16>) {
      ^bb0(%in: i16, %out: i16):
        linalg.yield %in : i16
      } -> tensor<16xi16>
      %cst_2 = arith.constant dense<0> : tensor<16x16xi16>
      %2 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%cst : tensor<16x16xi16>) outs(%cst_2 : tensor<16x16xi16>) {
      ^bb0(%in: i16, %out: i16):
        linalg.yield %in : i16
      } -> tensor<16x16xi16>
      %c0_i64 = arith.constant 0 : i64
      %3 = tensor_ext.rotate %input0, %c0_i64 : tensor<16xi16>, i64
      %extracted_slice = tensor.extract_slice %2[0, 0] [1, 16] [1, 1] : tensor<16x16xi16> to tensor<16xi16>
      %4 = arith.muli %3, %extracted_slice : tensor<16xi16>
      %5 = arith.addi %1, %4 : tensor<16xi16>
      %c1_i64 = arith.constant 1 : i64
      %6 = tensor_ext.rotate %input0, %c1_i64 : tensor<16xi16>, i64
      %extracted_slice_3 = tensor.extract_slice %2[1, 0] [1, 16] [1, 1] : tensor<16x16xi16> to tensor<16xi16>
      %7 = arith.muli %6, %extracted_slice_3 : tensor<16xi16>
      %8 = arith.addi %5, %7 : tensor<16xi16>
      %c2_i64 = arith.constant 2 : i64
      %9 = tensor_ext.rotate %input0, %c2_i64 : tensor<16xi16>, i64
      %extracted_slice_4 = tensor.extract_slice %2[2, 0] [1, 16] [1, 1] : tensor<16x16xi16> to tensor<16xi16>
      %10 = arith.muli %9, %extracted_slice_4 : tensor<16xi16>
      %11 = arith.addi %8, %10 : tensor<16xi16>
      %c3_i64 = arith.constant 3 : i64
      %12 = tensor_ext.rotate %input0, %c3_i64 : tensor<16xi16>, i64
      %extracted_slice_5 = tensor.extract_slice %2[3, 0] [1, 16] [1, 1] : tensor<16x16xi16> to tensor<16xi16>
      %13 = arith.muli %12, %extracted_slice_5 : tensor<16xi16>
      %14 = arith.addi %11, %13 : tensor<16xi16>
      %c4_i64 = arith.constant 4 : i64
      %15 = tensor_ext.rotate %input0, %c4_i64 : tensor<16xi16>, i64
      %extracted_slice_6 = tensor.extract_slice %2[4, 0] [1, 16] [1, 1] : tensor<16x16xi16> to tensor<16xi16>
      %16 = arith.muli %15, %extracted_slice_6 : tensor<16xi16>
      %17 = arith.addi %14, %16 : tensor<16xi16>
      %c5_i64 = arith.constant 5 : i64
      %18 = tensor_ext.rotate %input0, %c5_i64 : tensor<16xi16>, i64
      %extracted_slice_7 = tensor.extract_slice %2[5, 0] [1, 16] [1, 1] : tensor<16x16xi16> to tensor<16xi16>
      %19 = arith.muli %18, %extracted_slice_7 : tensor<16xi16>
      %20 = arith.addi %17, %19 : tensor<16xi16>
      %c6_i64 = arith.constant 6 : i64
      %21 = tensor_ext.rotate %input0, %c6_i64 : tensor<16xi16>, i64
      %extracted_slice_8 = tensor.extract_slice %2[6, 0] [1, 16] [1, 1] : tensor<16x16xi16> to tensor<16xi16>
      %22 = arith.muli %21, %extracted_slice_8 : tensor<16xi16>
      %23 = arith.addi %20, %22 : tensor<16xi16>
      %c7_i64 = arith.constant 7 : i64
      %24 = tensor_ext.rotate %input0, %c7_i64 : tensor<16xi16>, i64
      %extracted_slice_9 = tensor.extract_slice %2[7, 0] [1, 16] [1, 1] : tensor<16x16xi16> to tensor<16xi16>
      %25 = arith.muli %24, %extracted_slice_9 : tensor<16xi16>
      %26 = arith.addi %23, %25 : tensor<16xi16>
      %c8_i64 = arith.constant 8 : i64
      %27 = tensor_ext.rotate %input0, %c8_i64 : tensor<16xi16>, i64
      %extracted_slice_10 = tensor.extract_slice %2[8, 0] [1, 16] [1, 1] : tensor<16x16xi16> to tensor<16xi16>
      %28 = arith.muli %27, %extracted_slice_10 : tensor<16xi16>
      %29 = arith.addi %26, %28 : tensor<16xi16>
      %c9_i64 = arith.constant 9 : i64
      %30 = tensor_ext.rotate %input0, %c9_i64 : tensor<16xi16>, i64
      %extracted_slice_11 = tensor.extract_slice %2[9, 0] [1, 16] [1, 1] : tensor<16x16xi16> to tensor<16xi16>
      %31 = arith.muli %30, %extracted_slice_11 : tensor<16xi16>
      %32 = arith.addi %29, %31 : tensor<16xi16>
      %c10_i64 = arith.constant 10 : i64
      %33 = tensor_ext.rotate %input0, %c10_i64 : tensor<16xi16>, i64
      %extracted_slice_12 = tensor.extract_slice %2[10, 0] [1, 16] [1, 1] : tensor<16x16xi16> to tensor<16xi16>
      %34 = arith.muli %33, %extracted_slice_12 : tensor<16xi16>
      %35 = arith.addi %32, %34 : tensor<16xi16>
      %c11_i64 = arith.constant 11 : i64
      %36 = tensor_ext.rotate %input0, %c11_i64 : tensor<16xi16>, i64
      %extracted_slice_13 = tensor.extract_slice %2[11, 0] [1, 16] [1, 1] : tensor<16x16xi16> to tensor<16xi16>
      %37 = arith.muli %36, %extracted_slice_13 : tensor<16xi16>
      %38 = arith.addi %35, %37 : tensor<16xi16>
      %c12_i64 = arith.constant 12 : i64
      %39 = tensor_ext.rotate %input0, %c12_i64 : tensor<16xi16>, i64
      %extracted_slice_14 = tensor.extract_slice %2[12, 0] [1, 16] [1, 1] : tensor<16x16xi16> to tensor<16xi16>
      %40 = arith.muli %39, %extracted_slice_14 : tensor<16xi16>
      %41 = arith.addi %38, %40 : tensor<16xi16>
      %c13_i64 = arith.constant 13 : i64
      %42 = tensor_ext.rotate %input0, %c13_i64 : tensor<16xi16>, i64
      %extracted_slice_15 = tensor.extract_slice %2[13, 0] [1, 16] [1, 1] : tensor<16x16xi16> to tensor<16xi16>
      %43 = arith.muli %42, %extracted_slice_15 : tensor<16xi16>
      %44 = arith.addi %41, %43 : tensor<16xi16>
      %c14_i64 = arith.constant 14 : i64
      %45 = tensor_ext.rotate %input0, %c14_i64 : tensor<16xi16>, i64
      %extracted_slice_16 = tensor.extract_slice %2[14, 0] [1, 16] [1, 1] : tensor<16x16xi16> to tensor<16xi16>
      %46 = arith.muli %45, %extracted_slice_16 : tensor<16xi16>
      %47 = arith.addi %44, %46 : tensor<16xi16>
      %c15_i64 = arith.constant 15 : i64
      %48 = tensor_ext.rotate %input0, %c15_i64 : tensor<16xi16>, i64
      %extracted_slice_17 = tensor.extract_slice %2[15, 0] [1, 16] [1, 1] : tensor<16x16xi16> to tensor<16xi16>
      %49 = arith.muli %48, %extracted_slice_17 : tensor<16xi16>
      %50 = arith.addi %47, %49 : tensor<16xi16>
      secret.yield %50 : tensor<16xi16>
    } -> !secret.secret<tensor<16xi16>>
    return %0 : !secret.secret<tensor<16xi16>>
  }
}
