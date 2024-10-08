// RUN: heir-opt --secret-distribute-generic=distribute-through="affine.for" --yosys-optimizer="unroll-factor=2" --canonicalize %s | FileCheck %s

!in_ty = !secret.secret<memref<10xi8>>
!out_ty = !secret.secret<memref<10xi8>>

func.func @basic_example(%arg0: !in_ty) -> (!out_ty) {
  %0 = secret.generic {
  ^bb0:
    %memref = memref.alloc() : memref<10xi8>
    secret.yield %memref : memref<10xi8>
  } -> !out_ty

  affine.for %i = 0 to 10 {
    secret.generic ins(%arg0, %0 : !in_ty, !out_ty) {
    ^bb0(%clean_memref: memref<10xi8>, %clean_outmemref: memref<10xi8>):
      %1 = memref.load %clean_memref[%i] : memref<10xi8>
      // This is actually such a simple computation that yosys will optimize it
      // to be purely assignments (doubling a number is shifting the bits and
      // assigning 0 to the lowest bit).
      %2 = arith.addi %1, %1 : i8
      memref.store %2, %clean_outmemref[%i] : memref<10xi8>
      secret.yield
    }
  }

  return %0 : !out_ty
}

// CHECK-LABEL: func.func @basic_example(
// CHECK-SAME: %[[arg0:.*]]: [[secret_ty:!secret.secret<memref<10xi8>>]]
//   CHECK-DAG: %[[c0:.*]] = arith.constant 0
//   CHECK-DAG: %[[c1:.*]] = arith.constant 1
//   CHECK-DAG: %[[c2:.*]] = arith.constant 2
//   CHECK-DAG: %[[c3:.*]] = arith.constant 3
//   CHECK-DAG: %[[c4:.*]] = arith.constant 4
//   CHECK-DAG: %[[c5:.*]] = arith.constant 5
//   CHECK-DAG: %[[c6:.*]] = arith.constant 6
//   CHECK-DAG: %[[c7:.*]] = arith.constant 7
//   CHECK-DAG: %[[false:.*]] = arith.constant false
//
//   CHECK: secret.generic
//     CHECK-NEXT: memref.alloc
//     CHECK-NEXT: secret.yield
//
//   CHECK: affine.for %[[index:.*]] = 0 to 10 step 2 {
//   CHECK-NEXT:  %[[index_plus_one:.*]] = affine.apply
//
//                The loads are hoisted out of the generic
//   CHECK-NEXT:  secret.generic
//   CHECK-NEXT:  ^bb
//   CHECK-NEXT:    memref.load
//   CHECK-SAME:    %[[index]]
//   CHECK-NEXT:    secret.yield
//   CHECK-NEXT:  } -> !secret.secret<i8>
//   CHECK-NEXT:  secret.generic
//   CHECK-NEXT:  ^bb
//   CHECK-NEXT:    memref.load
//   CHECK-SAME:    %[[index_plus_one]]
//   CHECK-NEXT:    secret.yield
//   CHECK-NEXT:  } -> !secret.secret<i8>
//
//   CHECK-NEXT:  secret.cast
//   CHECK-NEXT:  secret.cast
//
//                The main computation
//   CHECK-NEXT:  secret.generic
//   CHECK-NEXT:  ^bb{{.*}}(%[[arg2:.*]]: memref<8xi1>, %[[arg3:.*]]: memref<8xi1>):
//                  Note bit 7 is never loaded because it is shifted out
//   CHECK-DAG:    %[[arg2bit0:.*]] = memref.load %arg2[%[[c0]]] : memref<8xi1>
//   CHECK-DAG:    %[[arg2bit1:.*]] = memref.load %arg2[%[[c1]]] : memref<8xi1>
//   CHECK-DAG:    %[[arg2bit2:.*]] = memref.load %arg2[%[[c2]]] : memref<8xi1>
//   CHECK-DAG:    %[[arg2bit3:.*]] = memref.load %arg2[%[[c3]]] : memref<8xi1>
//   CHECK-DAG:    %[[arg2bit4:.*]] = memref.load %arg2[%[[c4]]] : memref<8xi1>
//   CHECK-DAG:    %[[arg2bit5:.*]] = memref.load %arg2[%[[c5]]] : memref<8xi1>
//   CHECK-DAG:    %[[arg2bit6:.*]] = memref.load %arg2[%[[c6]]] : memref<8xi1>
//
//   CHECK-DAG:    %[[arg3bit0:.*]] = memref.load %arg3[%[[c0]]] : memref<8xi1>
//   CHECK-DAG:    %[[arg3bit1:.*]] = memref.load %arg3[%[[c1]]] : memref<8xi1>
//   CHECK-DAG:    %[[arg3bit2:.*]] = memref.load %arg3[%[[c2]]] : memref<8xi1>
//   CHECK-DAG:    %[[arg3bit3:.*]] = memref.load %arg3[%[[c3]]] : memref<8xi1>
//   CHECK-DAG:    %[[arg3bit4:.*]] = memref.load %arg3[%[[c4]]] : memref<8xi1>
//   CHECK-DAG:    %[[arg3bit5:.*]] = memref.load %arg3[%[[c5]]] : memref<8xi1>
//   CHECK-DAG:    %[[arg3bit6:.*]] = memref.load %arg3[%[[c6]]] : memref<8xi1>
//
//   The order of use of the two allocs seem arbitrary and nondeterministic,
//   so check the stores without the memref names
//   CHECK-DAG:    memref.alloc() : memref<8xi1>
//   CHECK-DAG:    memref.alloc() : memref<8xi1>
//   CHECK-DAG:    memref.store %[[false]], %{{.*}}[%[[c0]]] : memref<8xi1>
//   CHECK-DAG:    memref.store %[[arg3bit0]],  %{{.*}}[%[[c1]]] : memref<8xi1>
//   CHECK-DAG:    memref.store %[[arg3bit1]],  %{{.*}}[%[[c2]]] : memref<8xi1>
//   CHECK-DAG:    memref.store %[[arg3bit2]],  %{{.*}}[%[[c3]]] : memref<8xi1>
//   CHECK-DAG:    memref.store %[[arg3bit3]],  %{{.*}}[%[[c4]]] : memref<8xi1>
//   CHECK-DAG:    memref.store %[[arg3bit4]],  %{{.*}}[%[[c5]]] : memref<8xi1>
//   CHECK-DAG:    memref.store %[[arg3bit5]],  %{{.*}}[%[[c6]]] : memref<8xi1>
//   CHECK-DAG:    memref.store %[[arg3bit6]],  %{{.*}}[%[[c7]]] : memref<8xi1>
//   CHECK-DAG:    memref.store %[[false]], %{{.*}}[%[[c0]]] : memref<8xi1>
//   CHECK-DAG:    memref.store %[[arg2bit0]],  %{{.*}}[%[[c1]]] : memref<8xi1>
//   CHECK-DAG:    memref.store %[[arg2bit1]],  %{{.*}}[%[[c2]]] : memref<8xi1>
//   CHECK-DAG:    memref.store %[[arg2bit2]],  %{{.*}}[%[[c3]]] : memref<8xi1>
//   CHECK-DAG:    memref.store %[[arg2bit3]],  %{{.*}}[%[[c4]]] : memref<8xi1>
//   CHECK-DAG:    memref.store %[[arg2bit4]],  %{{.*}}[%[[c5]]] : memref<8xi1>
//   CHECK-DAG:    memref.store %[[arg2bit5]],  %{{.*}}[%[[c6]]] : memref<8xi1>
//   CHECK-DAG:    memref.store %[[arg2bit6]],  %{{.*}}[%[[c7]]] : memref<8xi1>
//
//   CHECK-NEXT:    secret.yield {{.*}}, {{.*}}
//   CHECK-NEXT:  }
//   CHECK-NEXT:  secret.cast
//   CHECK-NEXT:  secret.cast
//   CHECK-NEXT:  secret.generic
//   CHECK-NEXT:  ^bb
//   CHECK-NEXT:    memref.store
//   CHECK-NEXT:    secret.yield
//   CHECK-NEXT:  }
//   CHECK-NEXT:  secret.generic
//   CHECK-NEXT:  ^bb
//   CHECK-NEXT:    memref.store
//   CHECK-NEXT:    secret.yield
//   CHECK-NEXT:  }
//   CHECK-NEXT:}
//   CHECK-NEXT: return

// Computes the set of partial cumulative sums of the input array
func.func @cumulative_sums(%arg0: !in_ty) -> (!out_ty) {
  %0 = secret.generic {
  ^bb0:
    %memref = memref.alloc() : memref<10xi8>
    secret.yield %memref : memref<10xi8>
  } -> !out_ty

  secret.generic ins(%arg0, %0 : !in_ty, !out_ty) {
  ^bb0(%input: memref<10xi8>, %alloc: memref<10xi8>):
    %c0 = arith.constant 0 : index
    %val = memref.load %input[%c0] : memref<10xi8>
    memref.store %val, %alloc[%c0] : memref<10xi8>
    secret.yield
  }

  affine.for %i = 1 to 10 {
    secret.generic ins(%arg0, %0 : !in_ty, !out_ty) {
    ^bb0(%input: memref<10xi8>, %accum: memref<10xi8>):
      %c1 = arith.constant 1 : index
      %i_minus_one = arith.subi %i, %c1 : index
      %next_val = memref.load %input[%i] : memref<10xi8>
      %prev_sum = memref.load %accum[%i_minus_one] : memref<10xi8>
      %next_sum  = arith.addi %prev_sum, %next_val : i8
      memref.store %next_sum, %accum[%i] : memref<10xi8>
      secret.yield
    }
  }

  return %0 : !out_ty
}

// CHECK-LABEL: func.func @cumulative_sums
// CHECK-SAME: %[[arg0:.*]]: [[secret_ty:!secret.secret<memref<10xi8>>]]
//   CHECK-DAG: %[[c0:.*]] = arith.constant 0
//   CHECK-DAG: %[[c1:.*]] = arith.constant 1
//   CHECK-DAG: %[[c2:.*]] = arith.constant 2
//   CHECK-DAG: %[[c3:.*]] = arith.constant 3
//   CHECK-DAG: %[[c4:.*]] = arith.constant 4
//   CHECK-DAG: %[[c5:.*]] = arith.constant 5
//   CHECK-DAG: %[[c6:.*]] = arith.constant 6
//   CHECK-DAG: %[[c7:.*]] = arith.constant 7
//   CHECK-DAG: %[[false:.*]] = arith.constant false
//
//   Extracting the initial cumulative sum
//   CHECK: secret.generic ins(%[[arg0]]
//     CHECK-NEXT: bb
//     CHECK-NEXT: memref.load
//     CHECK-NEXT: secret.yield
//
//   Allocating storage for the output
//   CHECK: %[[output:.*]] = secret.generic
//     CHECK-NEXT: memref.alloc
//     CHECK-NEXT: secret.yield
//
//   Storing the initial cumulative sum
//   CHECK: secret.generic
//     CHECK-NEXT: bb
//     CHECK-NEXT: memref.store
//     CHECK-NEXT: secret.yield
//
//   Main loop
//   CHECK: affine.for %[[index:.*]] = 1 to 9 step 2 {
//     CHECK-NEXT: affine.apply
//     Extracted load ops from main loop body
//     CHECK-NEXT: %[[load0:.*]] = secret.generic
//       CHECK-NEXT: bb0
//       CHECK-NEXT: memref.load
//       CHECK-NEXT: secret.yield
//     CHECK-NEXT: }
//     CHECK-NEXT: %[[load1:.*]] = secret.generic
//       CHECK-NEXT: bb0
//       CHECK-NEXT: memref.load
//       CHECK-NEXT: secret.yield
//     CHECK-NEXT: }
//
//     Extracted plaintext arith op
//     CHECK-NEXT: %[[index_minus_one:.*]] = arith.subi %[[index]], %[[c1]]
//     Same deal, but for second unwrapped loop iteration marked by SECOND_SUB
//     CHECK-NEXT: arith.subi
//
//     Extracted load that can only be extracted because the previous
//     arith op was extracted.
//     CHECK-NEXT: secret.generic
//       CHECK-NEXT: bb
//       CHECK-NEXT: memref.load
//       CHECK-SAME: %[[index_minus_one]]
//       CHECK-NEXT: secret.yield
//     CHECK-NEXT: }
//
//     mark: SECOND_SUB
//     CHECK-NEXT: secret.generic
//       CHECK-NEXT: bb
//       CHECK-NEXT: memref.load
//       CHECK-NEXT: secret.yield
//     CHECK-NEXT: }
//
//     Main secret body
//     CHECK-COUNT-4: secret.cast
//     CHECK-NEXT: secret.generic
//       CHECK-NEXT: bb
//       CHECK-COUNT-30: comb.truth_table
//       // for the output of the generic
//       CHECK: memref.alloc
//       CHECK-COUNT-8: memref.store
//       CHECK: memref.alloc
//       CHECK-COUNT-8: memref.store
//       CHECK: secret.yield %[[ret0:.*]], %[[ret1:.*]]
//
//     Output casts
//     CHECK-COUNT-2: secret.cast
//
//     Store generic outputs in output memref
//     CHECK: secret.generic
//       CHECK-NEXT: bb
//       CHECK-NEXT: memref.store
//       CHECK-NEXT: secret.yield
//     CHECK: secret.generic
//       CHECK-NEXT: bb
//       CHECK-NEXT: memref.store
//       CHECK-NEXT: secret.yield
//   }
//
//   Because there are an odd number of iterations and we unroll by a factor of
//   2, the last iteration is peeled off into its own section. Two loads,
//   followed by two casts, a generic representing one loop iteration (not two,
//   so half the truth_table ops), and then output storing.
//
//   CHECK: %[[load0:.*]] = secret.generic
//     CHECK-NEXT: bb0
//     CHECK-NEXT: memref.load
//     CHECK-NEXT: secret.yield
//   CHECK: %[[load1:.*]] = secret.generic
//     CHECK-NEXT: bb0
//     CHECK-NEXT: memref.load
//     CHECK-NEXT: secret.yield
//   CHECK-COUNT-2: secret.cast
//   CHECK: secret.generic
//     CHECK-NEXT: bb
//     CHECK-COUNT-15: comb.truth_table
//     // for the output of the generic
//     CHECK: memref.alloc
//     CHECK-COUNT-8: memref.store
//     CHECK: secret.yield %[[ret2:.*]]
//
//   CHECK: secret.cast
//   CHECK: secret.generic
//     CHECK-NEXT: bb
//     CHECK-NEXT: memref.store
//     CHECK-NEXT: secret.yield
//
//   CHECK: return %[[output]]
