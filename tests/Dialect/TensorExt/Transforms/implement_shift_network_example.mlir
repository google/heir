// RUN: heir-opt --implement-shift-network %s | FileCheck %s

#map = dense<[
  [0, 0, 0, 13],
  [0, 1, 0, 8],
  [0, 2, 0, 4],
  [0, 3, 0, 0],
  [0, 4, 0, 11],
  [0, 5, 0, 7],
  [0, 6, 0, 14],
  [0, 7, 0, 5],
  [0, 8, 0, 15],
  [0, 9, 0, 3],
  [0, 10, 0, 12],
  [0, 11, 0, 6],
  [0, 12, 0, 10],
  [0, 13, 0, 2],
  [0, 14, 0, 9],
  [0, 15, 0, 1]]> : tensor<16x4xi64>
func.func @figure3(%0: tensor<1x16xi32>) -> tensor<1x16xi32> {
  %1 = tensor_ext.permute %0 {permutation = #map} : tensor<1x16xi32>
  return %1 : tensor<1x16xi32>
}

// CHECK: @figure3
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x16xi32>) -> tensor<1x16xi32>
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[ARG0]][0, 0] [1, 16] [1, 1] : tensor<1x16xi32> to tensor<16xi32>
// CHECK: %[[CST:.*]] = arith.constant dense<[1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]> : tensor<16xi32>
// CHECK: %[[MUL0:.*]] = arith.muli %[[SLICE]], %[[CST]] : tensor<16xi32>
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[ROT0:.*]] = tensor_ext.rotate %[[MUL0]], %[[C1]] : tensor<16xi32>, index
// CHECK: %[[CST_0:.*]] = arith.constant dense<[1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]> : tensor<16xi32>
// CHECK: %[[MUL1:.*]] = arith.muli %[[ROT0]], %[[CST_0]] : tensor<16xi32>
// CHECK: %[[CST_1:.*]] = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]> : tensor<16xi32>
// CHECK: %[[MUL2:.*]] = arith.muli %[[ROT0]], %[[CST_1]] : tensor<16xi32>
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[ROT1:.*]] = tensor_ext.rotate %[[MUL2]], %[[C2]] : tensor<16xi32>, index
// CHECK: %[[ADD0:.*]] = arith.addi %[[MUL1]], %[[ROT1]] : tensor<16xi32>
// CHECK: %[[CST_2:.*]] = arith.constant dense<[1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]> : tensor<16xi32>
// CHECK: %[[MUL3:.*]] = arith.muli %[[ADD0]], %[[CST_2]] : tensor<16xi32>
// CHECK: %[[CST_3:.*]] = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]> : tensor<16xi32>
// CHECK: %[[MUL4:.*]] = arith.muli %[[ADD0]], %[[CST_3]] : tensor<16xi32>
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[ROT2:.*]] = tensor_ext.rotate %[[MUL4]], %[[C4]] : tensor<16xi32>, index
// CHECK: %[[ADD1:.*]] = arith.addi %[[MUL3]], %[[ROT2]] : tensor<16xi32>
// CHECK: %[[CST_4:.*]] = arith.constant dense<[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]> : tensor<16xi32>
// CHECK: %[[MUL5:.*]] = arith.muli %[[ADD1]], %[[CST_4]] : tensor<16xi32>
// CHECK: %[[CST_5:.*]] = arith.constant dense<[1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi32>
// CHECK: %[[MUL6:.*]] = arith.muli %[[ADD1]], %[[CST_5]] : tensor<16xi32>
// CHECK: %[[C8:.*]] = arith.constant 8 : index
// CHECK: %[[ROT3:.*]] = tensor_ext.rotate %[[MUL6]], %[[C8]] : tensor<16xi32>, index
// CHECK: %[[ADD2:.*]] = arith.addi %[[MUL5]], %[[ROT3]] : tensor<16xi32>
// CHECK: %[[CST_6:.*]] = arith.constant dense<[0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0]> : tensor<16xi32>
// CHECK: %[[MUL7:.*]] = arith.muli %[[SLICE]], %[[CST_6]] : tensor<16xi32>
// CHECK: %[[CST_7:.*]] = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]> : tensor<16xi32>
// CHECK: %[[MUL8:.*]] = arith.muli %[[SLICE]], %[[CST_7]] : tensor<16xi32>
// CHECK: %[[C1_8:.*]] = arith.constant 1 : index
// CHECK: %[[ROT4:.*]] = tensor_ext.rotate %[[MUL8]], %[[C1_8]] : tensor<16xi32>, index
// CHECK: %[[ADD3:.*]] = arith.addi %[[MUL7]], %[[ROT4]] : tensor<16xi32>
// CHECK: %[[CST_9:.*]] = arith.constant dense<[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]> : tensor<16xi32>
// CHECK: %[[MUL9:.*]] = arith.muli %[[ADD3]], %[[CST_9]] : tensor<16xi32>
// CHECK: %[[CST_10:.*]] = arith.constant dense<[0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0]> : tensor<16xi32>
// CHECK: %[[MUL10:.*]] = arith.muli %[[ADD3]], %[[CST_10]] : tensor<16xi32>
// CHECK: %[[C2_11:.*]] = arith.constant 2 : index
// CHECK: %[[ROT5:.*]] = tensor_ext.rotate %[[MUL10]], %[[C2_11]] : tensor<16xi32>, index
// CHECK: %[[ADD4:.*]] = arith.addi %[[MUL9]], %[[ROT5]] : tensor<16xi32>
// CHECK: %[[CST_12:.*]] = arith.constant dense<[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]> : tensor<16xi32>
// CHECK: %[[MUL11:.*]] = arith.muli %[[ADD4]], %[[CST_12]] : tensor<16xi32>
// CHECK: %[[CST_13:.*]] = arith.constant dense<[1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0]> : tensor<16xi32>
// CHECK: %[[MUL12:.*]] = arith.muli %[[ADD4]], %[[CST_13]] : tensor<16xi32>
// CHECK: %[[C4_14:.*]] = arith.constant 4 : index
// CHECK: %[[ROT6:.*]] = tensor_ext.rotate %[[MUL12]], %[[C4_14]] : tensor<16xi32>, index
// CHECK: %[[ADD5:.*]] = arith.addi %[[MUL11]], %[[ROT6]] : tensor<16xi32>
// CHECK: %[[CST_15:.*]] = arith.constant dense<[0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]> : tensor<16xi32>
// CHECK: %[[MUL13:.*]] = arith.muli %[[ADD5]], %[[CST_15]] : tensor<16xi32>
// CHECK: %[[CST_16:.*]] = arith.constant dense<[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1]> : tensor<16xi32>
// CHECK: %[[MUL14:.*]] = arith.muli %[[ADD5]], %[[CST_16]] : tensor<16xi32>
// CHECK: %[[C8_17:.*]] = arith.constant 8 : index
// CHECK: %[[ROT7:.*]] = tensor_ext.rotate %[[MUL14]], %[[C8_17]] : tensor<16xi32>, index
// CHECK: %[[ADD6:.*]] = arith.addi %[[MUL13]], %[[ROT7]] : tensor<16xi32>
// CHECK: %[[ADD7:.*]] = arith.addi %[[ADD2]], %[[ADD6]] : tensor<16xi32>
// CHECK: %[[CST_18:.*]] = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]> : tensor<16xi32>
// CHECK: %[[MUL15:.*]] = arith.muli %[[SLICE]], %[[CST_18]] : tensor<16xi32>
// CHECK: %[[CST_19:.*]] = arith.constant dense<[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]> : tensor<16xi32>
// CHECK: %[[MUL16:.*]] = arith.muli %[[SLICE]], %[[CST_19]] : tensor<16xi32>
// CHECK: %[[C1_20:.*]] = arith.constant 1 : index
// CHECK: %[[ROT8:.*]] = tensor_ext.rotate %[[MUL16]], %[[C1_20]] : tensor<16xi32>, index
// CHECK: %[[ADD8:.*]] = arith.addi %[[MUL15]], %[[ROT8]] : tensor<16xi32>
// CHECK: %[[CST_21:.*]] = arith.constant dense<[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]> : tensor<16xi32>
// CHECK: %[[MUL17:.*]] = arith.muli %[[ADD8]], %[[CST_21]] : tensor<16xi32>
// CHECK: %[[C2_22:.*]] = arith.constant 2 : index
// CHECK: %[[ROT9:.*]] = tensor_ext.rotate %[[MUL17]], %[[C2_22]] : tensor<16xi32>, index
// CHECK: %[[CST_23:.*]] = arith.constant dense<[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]> : tensor<16xi32>
// CHECK: %[[MUL18:.*]] = arith.muli %[[ROT9]], %[[CST_23]] : tensor<16xi32>
// CHECK: %[[CST_24:.*]] = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]> : tensor<16xi32>
// CHECK: %[[MUL19:.*]] = arith.muli %[[ROT9]], %[[CST_24]] : tensor<16xi32>
// CHECK: %[[C4_25:.*]] = arith.constant 4 : index
// CHECK: %[[ROT10:.*]] = tensor_ext.rotate %[[MUL19]], %[[C4_25]] : tensor<16xi32>, index
// CHECK: %[[ADD9:.*]] = arith.addi %[[MUL18]], %[[ROT10]] : tensor<16xi32>
// CHECK: %[[CST_26:.*]] = arith.constant dense<[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi32>
// CHECK: %[[MUL20:.*]] = arith.muli %[[ADD9]], %[[CST_26]] : tensor<16xi32>
// CHECK: %[[CST_27:.*]] = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]> : tensor<16xi32>
// CHECK: %[[MUL21:.*]] = arith.muli %[[ADD9]], %[[CST_27]] : tensor<16xi32>
// CHECK: %[[C8_28:.*]] = arith.constant 8 : index
// CHECK: %[[ROT11:.*]] = tensor_ext.rotate %[[MUL21]], %[[C8_28]] : tensor<16xi32>, index
// CHECK: %[[ADD10:.*]] = arith.addi %[[MUL20]], %[[ROT11]] : tensor<16xi32>
// CHECK: %[[ADD11:.*]] = arith.addi %[[ADD7]], %[[ADD10]] : tensor<16xi32>
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1x16xi32>
// CHECK: %[[INSERT:.*]] = tensor.insert_slice %[[ADD11]] into %[[EMPTY]][0, 0] [1, 16] [1, 1] : tensor<16xi32> into tensor<1x16xi32>
// CHECK: return %[[INSERT]] : tensor<1x16xi32>
