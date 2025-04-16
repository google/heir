// RUN: heir-opt --mlir-to-cggi=abc-fast=true %s | FileCheck %s

// Calculate Variance of i4 elements, returning i32
// Uses the formula: Var(X) = E[X^2] - (E[X])^2 = (sum_sq / N) - (sum / N)^2
// CHECK: func.func @calculate_variance
func.func @calculate_variance(%input_tensor: tensor<5xi4> {secret.secret}) -> i32 {
    %c10_i32 = arith.constant 10 : i32

    // Initialize accumulators for sum and sum of squares
    %sum_init = arith.constant 0 : i32
    %sum_sq_init = arith.constant 0 : i32

    // Loop through the tensor to calculate sum and sum of squares
    %loop_result:2 = affine.for %iv = 0 to 5 iter_args(%sum_iter = %sum_init, %sum_sq_iter = %sum_sq_init) -> (i32, i32) {
        // Extract element
        %element_i4 = tensor.extract %input_tensor[%iv] : tensor<5xi4>
        // Extend to i32
        %element_i32 = arith.extsi %element_i4 : i4 to i32

        // Accumulate sum
        %next_sum = arith.addi %sum_iter, %element_i32 : i32

        // Calculate square
        %element_sq_i32 = arith.muli %element_i32, %element_i32 : i32
        // Accumulate sum of squares
        %next_sum_sq = arith.addi %sum_sq_iter, %element_sq_i32 : i32

        affine.yield %next_sum, %next_sum_sq : i32, i32
    }

    // Calculate E[X] = sum / N (Use loop result #0 directly)
    %mean = arith.divsi %loop_result#0, %c10_i32 : i32

    // Calculate E[X^2] = sum_sq / N (Use loop result #1 directly)
    %mean_sq_val = arith.divsi %loop_result#1, %c10_i32 : i32

    // Calculate (E[X])^2 = mean * mean
    %mean_pow2 = arith.muli %mean, %mean : i32

    // Calculate Variance = E[X^2] - (E[X])^2
    %variance = arith.subi %mean_sq_val, %mean_pow2 : i32

    return %variance : i32
}
