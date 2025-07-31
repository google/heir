// RUN: heir-opt --mlir-to-cggi=abc-fast=true --scheme-to-tfhe-rs %s | FileCheck %s

// CHECK: module
module {
  func.func @main(%arg0: tensor<1x1xi8> {iree.identifier = "serving_default_dense_input:0", secret.secret, tf_saved_model.index_path = ["dense_input"]}) -> (tensor<1x1xi32> {iree.identifier = "StatefulPartitionedCall:0", tf_saved_model.index_path = ["dense_2"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %cst = arith.constant dense<[[9, 54, 57, 71, 104, 115, 98, 99, 64, 26, 127, 25, 82, 68, 95, 86]]> : tensor<1x16xi8>
    %cst_0 = arith.constant dense<[[0, 0, 5438, 5515, 1352, 1500, 4152, 84, 3396, 0, 1981, 5581, 0, 6964, 3407, 7217]]> : tensor<1x16xi16>
    %cst_1 = arith.constant dense<[[729, 1954, 610, 0, 241, 471, 35, 867, 571, 581, 4260, 3943, 591, 0, 889, 5103]]> : tensor<1x16xi32>
    %cst_2 = arith.constant dense<429> : tensor<1x1xi32>
    %c0_i8 = arith.constant 0 : i8
    %c0_i16 = arith.constant 0 : i16
    %c0_i32 = arith.constant 0 : i32
    %cst_3 = arith.constant dense<"0xF403FB10E42311DD25EFEAE2DB19E9081ADC27FE150A0CFB211C1A1FE71E2422EDD2DD140722F5FD1DE7FCE915E2E6260902EBDA0B24E0000A03D8D7150929211906DB1C041EF314DBE013CAF5FD000921F9E4F81B2707261D1600E206F30708F4F405F31A031711DC02C4DDD614160F24021AF1FEE6E5172003D8C61ADDE20BE0FF17EFEB03E8E70121EC13DCDA1EE021FAFCE20124EDF1FA18D9E709200D12EFEF24F3DEFFFA11E309FE0422D923F4BCF1120921C0DCEA372E0D3EFE0FD37FF7EF15E3E611E8020BD9190008E3DDDCF5D3EFE90BF82326F1E5200102F9F758FA271EEDECFCFB041A14D81413F714E519E1E4E303F10704160BD6C7EFEEFA26"> : tensor<16x16xi8>
    %cst_4 = arith.constant dense<[[39], [59], [39], [21], [28], [32], [34], [35], [15], [27], [59], [41], [18], [35], [7], [127]]> : tensor<16x1xi8>
    %2 = linalg.quantized_matmul ins(%arg0, %cst, %c0_i8, %c0_i8 : tensor<1x1xi8>, tensor<1x16xi8>, i8, i8) outs(%cst_0 : tensor<1x16xi16>) -> tensor<1x16xi16>
    %4 = linalg.quantized_matmul ins(%2, %cst_3, %c0_i16, %c0_i16 : tensor<1x16xi16>, tensor<16x16xi8>, i16, i16) outs(%cst_1 : tensor<1x16xi32>) -> tensor<1x16xi32>
    %7 = linalg.quantized_matmul ins(%4, %cst_4, %c0_i32, %c0_i32 : tensor<1x16xi32>, tensor<16x1xi8>, i32, i32) outs(%cst_2 : tensor<1x1xi32>) -> tensor<1x1xi32>
    // CHECK: return
    return %7 : tensor<1x1xi32>
  }
}
