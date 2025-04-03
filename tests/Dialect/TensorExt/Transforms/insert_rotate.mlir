// RUN: heir-opt --insert-rotate --canonicalize --cse %s | FileCheck %s

func.func @test_insert_rotation_for_add(%arg1: tensor<16xi32>) -> i32 {
    %c4 = arith.constant 4 : index
    %c11 = arith.constant 11 : index
    %c15 = arith.constant 15 : index

    // These two ops are rotated both to align with each other, and so that the
    // result aligns with the %c4 rotation in extracted_1.
    %extracted = tensor.extract %arg1[%c11] : tensor<16xi32>
    %extracted_0 = tensor.extract %arg1[%c15] : tensor<16xi32>
    %1 = arith.addi %extracted, %extracted_0 : i32

    %extracted_1 = tensor.extract %arg1[%c4] : tensor<16xi32>
    %2 = arith.addi %1, %extracted_1 : i32
    return %2 : i32
}

// CHECK: func @test_insert_rotation_for_add
// CHECK-SAME: (%[[arg0:.*]]: tensor<16xi32>) -> i32 {
// CHECK-NEXT:    %[[c11:.*]] = arith.constant 11
// CHECK-NEXT:    %[[c4:.*]] = arith.constant 4
// CHECK-NEXT:    %[[c7:.*]] = arith.constant 7
// CHECK-NEXT:    %[[v0:.*]] = tensor_ext.rotate %[[arg0]], %[[c7]] : tensor<16xi32>, index
// CHECK-NEXT:    %[[v1:.*]] = tensor_ext.rotate %[[arg0]], %[[c11]] : tensor<16xi32>, index
// CHECK-NEXT:    %[[v2:.*]] = arith.addi %[[v0]], %[[v1]] : tensor<16xi32>
// CHECK-NEXT:    %[[v3:.*]] = arith.addi %[[v2]], %[[arg0]] : tensor<16xi32>
// CHECK-NEXT:    %[[extracted:.*]] = tensor.extract %[[v3]][%[[c4]]] : tensor<16xi32>
// CHECK-NEXT:    return

// This test should be applying the new patterns InsertRotations_SplatRHSToTensor,
// but the splats themselves should be elided.
// CHECK: @test_splat
// CHECK-NOT: tensor_ext.splat
func.func @test_splat(%arg1: tensor<16xi32>) -> tensor<16xi32> {
    %c0 = arith.constant 1 : index
    %c1 = arith.constant 1 : index
    %c11 = arith.constant 11 : index
    %c12 = arith.constant 12 : index
    %c15 = arith.constant 15 : index
    %cst = arith.constant dense<[[2, 3], [4, 5]]> : tensor<2x2xi32>

    %extracted = tensor.extract %arg1[%c11] : tensor<16xi32>
    %extracted_0 = tensor.extract %cst[%c1, %c1] : tensor<2x2xi32>
    %1 = arith.muli %extracted, %extracted_0 : i32

    %extracted_1 = tensor.extract %arg1[%c12] : tensor<16xi32>
    %extracted_2 = tensor.extract %cst[%c0, %c1] : tensor<2x2xi32>
    %2 = arith.muli %extracted_1, %extracted_2 : i32

    %3 = arith.addi %1, %2 : i32

    %inserted = tensor.insert %3 into %arg1[%c1] : tensor<16xi32>
    return %inserted : tensor<16xi32>
}
