// RUN: heir-opt --heir-tosa-to-arith %s > %t
// RUN: FileCheck %s < %t

// This model was produced by flatbuffer_translate --tflite-flatbuffer-to-mlir \
// tensorflow/lite/micro/examples/hello_world/models/hello_world_int8.tflite | \
// iree-import-tflite --output-format=mlir-ir
// Variable length tensors were then manually updated to size 1.
//
// See https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world/models

// CHECK-LABEL: module
#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (0)>
module {
  // CHECK-LABEL: func.func @main
  // CHECK-NOT: memref.global
  // CHECK-NOT: memref.copy
  func.func @main(%arg0: tensor<1x1xi8> {iree.identifier = "serving_default_dense_input:0", secret.secret, tf_saved_model.index_path = ["dense_input"]}) -> (tensor<1x1xi8> {iree.identifier = "StatefulPartitionedCall:0", tf_saved_model.index_path = ["dense_2"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %c5_i32 = arith.constant 5 : i32
    %c127_i32 = arith.constant 127 : i32
    %c0_i32 = arith.constant 0 : i32
    %c-128_i32 = arith.constant -128 : i32
    %cst = arith.constant dense<429> : tensor<1xi32>
    %cst_0 = arith.constant dense<[-729, 1954, 610, 0, 241, -471, -35, -867, 571, 581, 4260, 3943, 591, 0, -889, -5103]> : tensor<16xi32>
    %cst_1 = arith.constant dense<[0, 0, -5438, -5515, -1352, -1500, -4152, -84, 3396, 0, 1981, -5581, 0, -6964, 3407, -7217]> : tensor<16xi32>
    %c1073741824_i64 = arith.constant 1073741824 : i64
    %c-1073741824_i64 = arith.constant -1073741824 : i64
    %cst_2 = arith.constant dense<[[-9, -54, 57, 71, 104, 115, 98, 99, 64, -26, 127, 25, -82, 68, 95, 86]]> : tensor<1x16xi8>
    %c2039655736_i64 = arith.constant 2039655736 : i64
    %c38_i64 = arith.constant 38 : i64
    %c137438953472_i64 = arith.constant 137438953472 : i64
    %cst_3 = arith.constant dense<"0xF403FB10E42311DD25EFEAE2DB19E9081ADC27FE150A0CFB211C1A1FE71E2422EDD2DD140722F5FD1DE7FCE915E2E6260902EBDA0B24E0000A03D8D7150929211906DB1C041EF314DBE013CAF5FD000921F9E4F81B2707261D1600E206F30708F4F405F31A031711DC02C4DDD614160F24021AF1FEE6E5172003D8C61ADDE20BE0FF17EFEB03E8E70121EC13DCDA1EE021FAFCE20124EDF1FA18D9E709200D12EFEF24F3DEFFFA11E309FE0422D923F4BCF1120921C0DCEA372E0D3EFE0FD37FF7EF15E3E611E8020BD9190008E3DDDCF5D3EFE90BF82326F1E5200102F9F758FA271EEDECFCFB041A14D81413F714E519E1E4E303F10704160BD6C7EFEEFA26"> : tensor<16x16xi8>
    %c1561796795_i64 = arith.constant 1561796795 : i64
    %c37_i64 = arith.constant 37 : i64
    %c68719476736_i64 = arith.constant 68719476736 : i64
    %cst_4 = arith.constant dense<[[-39], [59], [39], [21], [28], [-32], [-34], [-35], [15], [27], [-59], [-41], [18], [-35], [-7], [127]]> : tensor<16x1xi8>
    %c1630361836_i64 = arith.constant 1630361836 : i64
    %c36_i64 = arith.constant 36 : i64
    %c34359738368_i64 = arith.constant 34359738368 : i64
    %0 = tensor.empty() : tensor<1x16xi8>
    %1 = tensor.empty() : tensor<1x16xi32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_1 : tensor<16xi32>) outs(%1 : tensor<1x16xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<1x16xi32>
    %3 = linalg.quantized_matmul ins(%arg0, %cst_2, %c-128_i32, %c0_i32 : tensor<1x1xi8>, tensor<1x16xi8>, i32, i32) outs(%2 : tensor<1x16xi32>) -> tensor<1x16xi32>
    %4 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<1x16xi32>) outs(%0 : tensor<1x16xi8>) {
    ^bb0(%in: i32, %out: i8):
      %13 = arith.extsi %in : i32 to i64
      %14 = arith.muli %13, %c2039655736_i64 : i64
      %15 = arith.addi %14, %c137438953472_i64 : i64
      %16 = arith.cmpi sge, %in, %c0_i32 : i32
      %17 = arith.select %16, %c1073741824_i64, %c-1073741824_i64 : i64
      %18 = arith.addi %17, %15 : i64
      %19 = arith.shrsi %18, %c38_i64 : i64
      %20 = arith.trunci %19 : i64 to i32
      %21 = arith.addi %20, %c-128_i32 : i32
      %22 = arith.maxsi %21, %c-128_i32 : i32
      %23 = arith.minsi %22, %c127_i32 : i32
      %24 = arith.trunci %23 : i32 to i8
      linalg.yield %24 : i8
    } -> tensor<1x16xi8>
    %5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : tensor<16xi32>) outs(%1 : tensor<1x16xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<1x16xi32>
    %6 = linalg.quantized_matmul ins(%4, %cst_3, %c-128_i32, %c0_i32 : tensor<1x16xi8>, tensor<16x16xi8>, i32, i32) outs(%5 : tensor<1x16xi32>) -> tensor<1x16xi32>
    %7 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<1x16xi32>) outs(%0 : tensor<1x16xi8>) {
    ^bb0(%in: i32, %out: i8):
      %13 = arith.extsi %in : i32 to i64
      %14 = arith.muli %13, %c1561796795_i64 : i64
      %15 = arith.addi %14, %c68719476736_i64 : i64
      %16 = arith.cmpi sge, %in, %c0_i32 : i32
      %17 = arith.select %16, %c1073741824_i64, %c-1073741824_i64 : i64
      %18 = arith.addi %17, %15 : i64
      %19 = arith.shrsi %18, %c37_i64 : i64
      %20 = arith.trunci %19 : i64 to i32
      %21 = arith.addi %20, %c-128_i32 : i32
      %22 = arith.maxsi %21, %c-128_i32 : i32
      %23 = arith.minsi %22, %c127_i32 : i32
      %24 = arith.trunci %23 : i32 to i8
      linalg.yield %24 : i8
    } -> tensor<1x16xi8>
    %8 = tensor.empty() : tensor<1x1xi32>
    %9 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst : tensor<1xi32>) outs(%8 : tensor<1x1xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<1x1xi32>
    %10 = linalg.quantized_matmul ins(%7, %cst_4, %c-128_i32, %c0_i32 : tensor<1x16xi8>, tensor<16x1xi8>, i32, i32) outs(%9 : tensor<1x1xi32>) -> tensor<1x1xi32>
    %11 = tensor.empty() : tensor<1x1xi8>
    %12 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<1x1xi32>) outs(%11 : tensor<1x1xi8>) {
    ^bb0(%in: i32, %out: i8):
      %13 = arith.extsi %in : i32 to i64
      %14 = arith.muli %13, %c1630361836_i64 : i64
      %15 = arith.addi %14, %c34359738368_i64 : i64
      %16 = arith.cmpi sge, %in, %c0_i32 : i32
      %17 = arith.select %16, %c1073741824_i64, %c-1073741824_i64 : i64
      %18 = arith.addi %17, %15 : i64
      %19 = arith.shrsi %18, %c36_i64 : i64
      %20 = arith.trunci %19 : i64 to i32
      %21 = arith.addi %20, %c5_i32 : i32
      %22 = arith.maxsi %21, %c-128_i32 : i32
      %23 = arith.minsi %22, %c127_i32 : i32
      %24 = arith.trunci %23 : i32 to i8
      linalg.yield %24 : i8
    } -> tensor<1x1xi8>
    // CHECK: return
    return %12 : tensor<1x1xi8>
  }
}
