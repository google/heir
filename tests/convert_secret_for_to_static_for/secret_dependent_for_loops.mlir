// RUN: heir-opt --convert-secret-for-to-static-for %s | FileCheck %s

// CHECK-LABEL: @for_loop_with_data_dependent_upper_bound
func.func @for_loop_with_data_dependent_upper_bound(%arg0: !secret.secret<tensor<32xi16>>, %arg1: !secret.secret<index>) -> !secret.secret<i16> {
    // CHECK-NEXT: %[[C0_I16:.*]] = arith.constant 0 : i16
    // CHECK-NEXT: %[[RESULT:.*]] = secret.generic ins(%[[ARG0:.*]], %[[ARG1:.*]] : !secret.secret<tensor<32xi16>>, !secret.secret<index>) {
    // CHECK-NEXT:  ^bb0(%[[ARG2:.*]]: tensor<32xi16>, %[[ARG3:.*]]: index):
    // CHECK-NEXT:   %[[FOR:.*]] = affine.for %[[IV:.*]] = 0 to 42 iter_args(%[[ACC:.*]] = %[[C0_I16]]) -> (i16) {
    // CHECK-NEXT:   %[[COND:.*]] = arith.cmpi slt, %[[IV]], %[[ARG3]] : index
    // CHECK-NEXT:    %[[IF:.*]] = scf.if %[[COND]] -> (i16) {
    // CHECK-NEXT:      %[[EXTRACTED:.*]] = tensor.extract %[[ARG2]][%[[IV]]] : tensor<32xi16>
    // CHECK-NEXT:      %[[ADD:.*]] = arith.addi %[[EXTRACTED]], %[[ACC]] : i16
    // CHECK-NEXT:      scf.yield %[[ADD]] : i16
    // CHECK-NEXT:     } else {
    // CHECK-NEXT:      scf.yield %[[ACC]] : i16
    // CHECK-NEXT:     }
    // CHECK-NEXT:    affine.yield %[[IF]] : i16
    // CHECK-NEXT:  } {lower = 0 : i64, upper = 42 : i64}
    // CHECK-NEXT:  secret.yield %[[FOR]] : i16
    // CHECK-NEXT: } -> !secret.secret<i16>
    // CHECK-NEXT: return %[[RESULT]] : !secret.secret<i16>
    %c0 = arith.constant 0 : index
    %c0_i16 = arith.constant 0 : i16
    %c1 = arith.constant 1 : index
    %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<tensor<32xi16>>, !secret.secret<index>) {
    ^bb0(%arg2: tensor<32xi16>, %arg3: index):
      %1 = scf.for %arg4 = %c0 to %arg3 step %c1 iter_args(%arg5 = %c0_i16) -> (i16) {
        %extracted = tensor.extract %arg2[%arg4] : tensor<32xi16>
        %2 = arith.addi %extracted, %arg5 : i16
        scf.yield %2 : i16
      } {lower = 0, upper = 42}
      secret.yield %1 : i16
    } -> !secret.secret<i16>
    return %0 : !secret.secret<i16>
}

// CHECK-LABEL: @for_loop_with_data_dependent_lower_bound
func.func @for_loop_with_data_dependent_lower_bound(%arg0: !secret.secret<tensor<32xi16>>, %arg1: !secret.secret<index>) -> !secret.secret<i16> {
    // CHECK-NEXT: %[[C0_I16:.*]] = arith.constant 0 : i16
    // CHECK-NEXT: %[[RESULT:.*]] = secret.generic ins(%[[ARG0:.*]], %[[ARG1:.*]] : !secret.secret<tensor<32xi16>>, !secret.secret<index>) {
    // CHECK-NEXT:  ^bb0(%[[ARG2:.*]]: tensor<32xi16>, %[[ARG3:.*]]: index):
    // CHECK-NEXT:   %[[FOR:.*]] = affine.for %[[IV:.*]] = 0 to 32 iter_args(%[[ACC:.*]] = %[[C0_I16]]) -> (i16) {
    // CHECK-NEXT:   %[[COND:.*]] = arith.cmpi sge, %[[IV]], %[[ARG3]] : index
    // CHECK-NEXT:    %[[IF:.*]] = scf.if %[[COND]] -> (i16) {
    // CHECK-NEXT:      %[[EXTRACTED:.*]] = tensor.extract %[[ARG2]][%[[IV]]] : tensor<32xi16>
    // CHECK-NEXT:      %[[ADD:.*]] = arith.addi %[[EXTRACTED]], %[[ACC]] : i16
    // CHECK-NEXT:      scf.yield %[[ADD]] : i16
    // CHECK-NEXT:     } else {
    // CHECK-NEXT:      scf.yield %[[ACC]] : i16
    // CHECK-NEXT:     }
    // CHECK-NEXT:    affine.yield %[[IF]] : i16
    // CHECK-NEXT:  } {lower = 0 : i64, upper = 42 : i64}
    // CHECK-NEXT:  secret.yield %[[FOR]] : i16
    // CHECK-NEXT: } -> !secret.secret<i16>
    // CHECK-NEXT: return %[[RESULT]] : !secret.secret<i16>
    %c32 = arith.constant 32 : index
    %c0_i16 = arith.constant 0 : i16
    %c1 = arith.constant 1 : index
    %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<tensor<32xi16>>, !secret.secret<index>) {
    ^bb0(%arg2: tensor<32xi16>, %arg3: index):
      %1 = scf.for %arg4 = %arg3 to %c32 step %c1 iter_args(%arg5 = %c0_i16) -> (i16) {
        %extracted = tensor.extract %arg2[%arg4] : tensor<32xi16>
        %2 = arith.addi %extracted, %arg5 : i16
        scf.yield %2 : i16
      } {lower = 0, upper = 42}
      secret.yield %1 : i16
    } -> !secret.secret<i16>
    return %0 : !secret.secret<i16>
}

// CHECK-LABEL: @for_loop_with_data_dependent_upper_and_lower_bounds
func.func @for_loop_with_data_dependent_upper_and_lower_bounds(%arg0: !secret.secret<tensor<32xi16>>, %lower: !secret.secret<index>, %upper: !secret.secret<index>) -> !secret.secret<i16> {
    // CHECK-NEXT: %[[C0_I16:.*]] = arith.constant 0 : i16
    // CHECK-NEXT: %[[RESULT:.*]] = secret.generic ins(%[[ARG0:.*]], %[[LOWER:.*]], %[[UPPER:.*]] : !secret.secret<tensor<32xi16>>, !secret.secret<index>, !secret.secret<index>) {
    // CHECK-NEXT:  ^bb0(%[[ARG3:.*]]: tensor<32xi16>, %[[ARG4:.*]]: index, %[[ARG5:.*]]: index):
    // CHECK-NEXT:   %[[FOR:.*]] = affine.for %[[IV:.*]] = 0 to 42 iter_args(%[[ACC:.*]] = %[[C0_I16]]) -> (i16) {
    // CHECK-NEXT:   %[[CMPIL:.*]] = arith.cmpi sge, %[[IV]], %[[ARG4]] : index
    // CHECK-NEXT:   %[[CMPIU:.*]] = arith.cmpi slt, %[[IV]], %[[ARG5]] : index
    // CHECK-NEXT:   %[[ANDI:.*]] = arith.andi %[[CMPIL]], %[[CMPIU]] : i1
    // CHECK-NEXT:    %[[IF:.*]] = scf.if %[[ANDI]] -> (i16) {
    // CHECK-NEXT:      %[[EXTRACTED:.*]] = tensor.extract %[[ARG3]][%[[IV]]] : tensor<32xi16>
    // CHECK-NEXT:      %[[ADD:.*]] = arith.addi %[[EXTRACTED]], %[[ACC]] : i16
    // CHECK-NEXT:      scf.yield %[[ADD]] : i16
    // CHECK-NEXT:     } else {
    // CHECK-NEXT:      scf.yield %[[ACC]] : i16
    // CHECK-NEXT:     }
    // CHECK-NEXT:    affine.yield %[[IF]] : i16
    // CHECK-NEXT:  } {lower = 0 : i64, upper = 42 : i64}
    // CHECK-NEXT:  secret.yield %[[FOR]] : i16
    // CHECK-NEXT: } -> !secret.secret<i16>
    // CHECK-NEXT: return %[[RESULT]] : !secret.secret<i16>
    %c0_i16 = arith.constant 0 : i16
    %c1 = arith.constant 1 : index
    %0 = secret.generic ins(%arg0, %lower, %upper : !secret.secret<tensor<32xi16>>, !secret.secret<index>,  !secret.secret<index>) {
    ^bb0(%arg3: tensor<32xi16>, %arg4: index, %arg5: index):
      %1 = scf.for %arg6 = %arg4 to %arg5 step %c1 iter_args(%arg7 = %c0_i16) -> (i16) {
        %extracted = tensor.extract %arg3[%arg6] : tensor<32xi16>
        %2 = arith.addi %extracted, %arg7 : i16
        scf.yield %2 : i16
      } {lower = 0, upper = 42}
      secret.yield %1 : i16
    } -> !secret.secret<i16>
    return %0 : !secret.secret<i16>
}

// CHECK-LABEL: @for_loop_with_data_dependent_upper_bound_multiple_iter_args
func.func @for_loop_with_data_dependent_upper_bound_multiple_iter_args(%arg0: !secret.secret<tensor<32xi16>>, %arg1: !secret.secret<index>) -> !secret.secret<i16> {
    // CHECK: %[[C0_I16:.*]] = arith.constant 0 : i16
    // CHECK: %[[RESULT:.*]] = secret.generic ins(%[[ARG0:.*]], %[[ARG1:.*]] : !secret.secret<tensor<32xi16>>, !secret.secret<index>) {
    // CHECK:  ^bb0(%[[ARG2:.*]]: tensor<32xi16>, %[[ARG3:.*]]: index):
    // CHECK:   %[[FOR:.*]]:2 = affine.for %[[I:.*]] = 0 to 42 iter_args(%[[ARG5:.*]] = %[[C0_I16]], %[[ARG6:.*]] = %[[C0_I16]]) -> (i16, i16) {
    // CHECK:      %[[CMPIU:.*]] = arith.cmpi slt, %[[I]], %[[ARG3]] : index
    // CHECK:      %[[IF:.*]]:2 = scf.if %[[CMPIU]] -> (i16, i16) {
    // CHECK:        %[[EXTRACTED:.*]] = tensor.extract %[[ARG2]][%[[I]]] : tensor<32xi16>
    // CHECK:        %[[ADD:.*]] = arith.addi %[[EXTRACTED]], %[[ARG5]] : i16
    // CHECK:        %[[MUL:.*]] = arith.muli %[[EXTRACTED]], %[[ARG6]] : i16
    // CHECK:        scf.yield %[[ADD]], %[[MUL]] : i16, i16
    // CHECK:      } else {
    // CHECK:        scf.yield %[[ARG5]], %[[ARG6]] : i16, i16
    // CHECK:      }
    // CHECK:      affine.yield %[[IF]]#0, %[[IF]]#1 : i16, i16
    // CHECK:    } {lower = 0 : i64, upper = 42 : i64}
    // CHECK:      %[[OUT:.*]] = arith.addi %[[FOR]]#0, %[[FOR]]#1 : i16
    // CHECK:      secret.yield %[[OUT]] : i16
    // CHECK:    } -> !secret.secret<i16>
    // CHECK:    return %[[RESULT]] : !secret.secret<i16>
    %c0 = arith.constant 0 : index
    %c0_i16 = arith.constant 0 : i16
    %c1 = arith.constant 1 : index
    %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<tensor<32xi16>>, !secret.secret<index>) {
    ^bb0(%arg2: tensor<32xi16>, %arg3: index):
      %1, %2 = scf.for %arg4 = %c0 to %arg3 step %c1 iter_args(%arg5 = %c0_i16, %arg6 = %c0_i16) -> (i16, i16) {
        %extracted = tensor.extract %arg2[%arg4] : tensor<32xi16>
        %3 = arith.addi %extracted, %arg5 : i16
        %4 = arith.muli %extracted, %arg6 : i16
        scf.yield %3, %4 : i16, i16
      } {lower = 0, upper = 42}
      %5 = arith.addi %1, %2 : i16
      secret.yield %5 : i16
    } -> !secret.secret<i16>
    return %0 : !secret.secret<i16>
}

// CHECK-LABEL: @partial_sum_with_secret_threshold
func.func @partial_sum_with_secret_threshold(%secretInput :!secret.secret<tensor<16xi16>>, %secretIndex: !secret.secret<index>, %secretThreshold: !secret.secret<i16>) -> (!secret.secret<i16>, !secret.secret<i16>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i16
  // CHECK: %[[RESULT:.*]]:2 = secret.generic ins(%[[SECRET_INPUT:.*]], %[[SECRET_INDEX:.*]], %[[SECRET_THRESHOLD:.*]] : !secret.secret<tensor<16xi16>>, !secret.secret<index>, !secret.secret<i16>) {
  // CHECK:  ^bb0(%[[INPUT:.*]]: tensor<16xi16>, %[[INDEX:.*]]: index, %[[THRESHOLD:.*]]: i16):
  // CHECK:   %[[FOR:.*]]:2 = affine.for %[[I:.*]] = 0 to 16 iter_args(%[[ARG1:.*]] = %[[C0]], %[[ARG2:.*]] = %[[C0]]) -> (i16, i16) {
  // CHECK:     %[[CMPI:.*]] = arith.cmpi slt, %[[I]], %[[INDEX]] : index
  // CHECK:     %[[IF:.*]]:2 = scf.if %[[CMPI]] -> (i16, i16) {
  // CHECK:        %[[COND:.*]] = arith.cmpi slt, %[[ARG1]], %[[THRESHOLD]] : i16
  // CHECK:        %[[EXTRACTED:.*]] = tensor.extract %[[INPUT]][%[[I]]] : tensor<16xi16>
  // CHECK:        %[[IFF:.*]]:2 = scf.if %[[COND]] -> (i16, i16) {
  // CHECK:          %[[SUM:.*]] = arith.addi %[[ARG1]], %[[EXTRACTED]] : i16
  // CHECK:          scf.yield %[[SUM]], %[[ARG2]] : i16, i16
  // CHECK:          } else {
  // CHECK:            %[[SUM:.*]] = arith.addi %[[ARG2]], %[[EXTRACTED]] : i16
  // CHECK:            scf.yield %[[ARG1]], %[[SUM]] : i16, i16
  // CHECK:           }
  // CHECK:        scf.yield %[[IFF]]#0, %[[IFF]]#1 : i16, i16
  // CHECK:       } else {
  // CHECK:        scf.yield %[[ARG1]], %[[ARG2]] : i16, i16
  // CHECK:      }
  // CHECK:        affine.yield %[[IF]]#0, %[[IF]]#1 : i16, i16
  // CHECK:     } {lower = 0 : i64, upper = 16 : i64}
  // CHECK:    secret.yield %[[FOR]]#0, %[[FOR]]#1 : i16, i16
  // CHECK: } -> (!secret.secret<i16>, !secret.secret<i16>)
  // CHECK: return %[[RESULT]]#0, %[[RESULT]]#1 : !secret.secret<i16>, !secret.secret<i16>
  // CHECK: }
  %start = arith.constant 0 : index
  %step = arith.constant 1 : index
  %c0 = arith.constant 0 : i16
  %0, %1 = secret.generic ins(%secretInput, %secretIndex, %secretThreshold : !secret.secret<tensor<16xi16>>, !secret.secret<index>, !secret.secret<i16>) {
  ^bb0(%input: tensor<16xi16>, %index: index, %threshold: i16):
      %2, %3 = scf.for %i = %start to %index step %step iter_args(%arg1 = %c0, %arg2 = %c0) -> (i16, i16) {
          %cond = arith.cmpi slt, %arg1, %threshold : i16
          %extracted = tensor.extract %input[%i] : tensor<16xi16>
          %sum1, %sum2 = scf.if %cond -> (i16, i16) {
              %sum = arith.addi %arg1, %extracted : i16
              scf.yield %sum, %arg2 : i16, i16
          } else {
              %sum = arith.addi %arg2, %extracted : i16
              scf.yield %arg1, %sum : i16, i16
          }
          scf.yield %sum1, %sum2 : i16, i16
      } {lower = 0, upper = 16}
      secret.yield %2, %3 : i16, i16
  }-> (!secret.secret<i16>, !secret.secret<i16>)
  return %0, %1 : !secret.secret<i16>, !secret.secret<i16>
}
