// RUN: heir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

#from = #rotom.layout<n = 16, dims = [[0:4:1], [1:4:1]]>
#to = #rotom.layout<n = 16, dims = [[1:4:1], [0:4:1]]>

// CHECK: rotom.convert_layout
// CHECK-SAME: from #{{.*}} to #
func.func @convert(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = rotom.convert_layout %arg0 : tensor<4x4xf32> from #from to #to
  return %0 : tensor<4x4xf32>
}

// -----

#from_n8 = #rotom.layout<n = 8, dims = [[0:4:1], [1:2:1]]>
#to_n16 = #rotom.layout<n = 16, dims = [[0:4:1], [1:4:1]]>

func.func @mismatched_n(%arg0: tensor<4x2xf32>) -> tensor<4x2xf32> {
  // expected-error@+1 {{the from and to layouts must have the same ciphertext size n; got 8 and 16}}
  %0 = rotom.convert_layout %arg0 : tensor<4x2xf32> from #from_n8 to #to_n16
  return %0 : tensor<4x2xf32>
}

// -----

// apply_roll may swap the rolled piece with its partner: the replicate-then-
// roll form, where a ciphertext replication trades places with a slot piece
// that then rolls by it.
#swap_from = #rotom.layout<n = 16, dims = [[R:4:1] | [1:4:1], [0:4:1]]>
#swap_to = #rotom.layout<n = 16, rolls = [(0, 1)], dims = [[1:4:1] | [R:4:1], [0:4:1]]>
// CHECK: rotom.apply_roll
func.func @apply_roll_swap(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = rotom.apply_roll %arg0 : tensor<4x4xf32> from #swap_from to #swap_to
  return %0 : tensor<4x4xf32>
}

// -----

#pieces_from = #rotom.layout<n = 16, dims = [[0:4:1], [1:4:1]]>
#pieces_to = #rotom.layout<n = 16, rolls = [(0, 1)], dims = [[0:4:1], [1:2:2], [1:2:1]]>
func.func @apply_roll_pieces_differ(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // expected-error@+1 {{the from and to layouts must have the same pieces}}
  %0 = rotom.apply_roll %arg0 : tensor<4x4xf32> from #pieces_from to #pieces_to
  return %0 : tensor<4x4xf32>
}

// -----

// A bsgs_matmul names the operand that arrives unrolled, the layout the roll
// would have produced, and the baby-step/giant-step split of its targets.
#bsgs = #rotom.layout<n = 16, dims = [[0:4:1], [1:4:1]]>
#bsgs_rolled = #rotom.layout<n = 16, rolls = [(0, 1)], dims = [[0:4:1], [1:4:1]]>
// CHECK: rotom.bsgs_matmul
// CHECK-SAME: roll_operand = 1
func.func @bsgs_matmul(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = rotom.bsgs_matmul %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    {lhs_layout = #bsgs, rhs_layout = #bsgs, rolled = #bsgs_rolled,
     compute = #bsgs, to = #bsgs, roll_operand = 1 : i64, roll_stride = 4 : i64,
     roll_targets = 4 : i64, baby = 2 : i64}
  return %0 : tensor<4x4xf32>
}

// -----

#bsgs_bad = #rotom.layout<n = 16, dims = [[0:4:1], [1:4:1]]>
#bsgs_bad_rolled = #rotom.layout<n = 16, rolls = [(0, 1)], dims = [[0:4:1], [1:4:1]]>
func.func @bsgs_matmul_baby_too_large(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // expected-error@+1 {{baby extent exceeds the target count}}
  %0 = rotom.bsgs_matmul %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    {lhs_layout = #bsgs_bad, rhs_layout = #bsgs_bad, rolled = #bsgs_bad_rolled,
     compute = #bsgs_bad, to = #bsgs_bad, roll_operand = 1 : i64,
     roll_stride = 4 : i64, roll_targets = 4 : i64, baby = 8 : i64}
  return %0 : tensor<4x4xf32>
}
