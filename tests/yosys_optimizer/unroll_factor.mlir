// RUN: heir-opt --secret-distribute-generic=distribute-through="affine.for" --yosys-optimizer="unroll-factor=3" --canonicalize %s | FileCheck %s

// Regression test for #444 testing the RTLIL imported through an unroll factor
// larger than the loop size.

!in_ty = !secret.secret<memref<2xi8>>
!out_ty = !secret.secret<memref<2xi8>>

func.func @basic_example(%arg0: !in_ty) -> (!out_ty) {
  %0 = secret.generic {
  ^bb0:
    %memref = memref.alloc() : memref<2xi8>
    secret.yield %memref : memref<2xi8>
  } -> !out_ty

  affine.for %i = 0 to 2 {
    secret.generic ins(%arg0, %0 : !in_ty, !out_ty) {
    ^bb0(%clean_memref: memref<2xi8>, %clean_outmemref: memref<2xi8>):
      %1 = memref.load %clean_memref[%i] : memref<2xi8>
      // This is actually such a simple computation that yosys will optimize it
      // to be purely assignments (doubling a number is shifting the bits and
      // assigning 0 to the lowest bit).
      %2 = arith.addi %1, %1 : i8
      memref.store %2, %clean_outmemref[%i] : memref<2xi8>
      secret.yield
    }
  }

  return %0 : !out_ty
}

// CHECK-LABEL: func.func @basic_example(
// CHECK-SAME: %[[arg0:.*]]: [[secret_ty:!secret.secret<memref<2xi8>>]]
//   CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[c2:.*]] = arith.constant 2 : index
//   CHECK-DAG: %[[c3:.*]] = arith.constant 3 : index
//   CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
//   CHECK-DAG: %[[c5:.*]] = arith.constant 5 : index
//   CHECK-DAG: %[[c6:.*]] = arith.constant 6 : index
//   CHECK-DAG: %[[c7:.*]] = arith.constant 7 : index
//   CHECK-DAG: %[[false:.*]] = arith.constant false
//
//   CHECK: secret.generic
//   CHECK-NEXT:  ^bb
//     CHECK-NEXT: memref.load
//     CHECK-NEXT: secret.yield
//   CHECK: secret.generic
//   CHECK-NEXT:  ^bb
//     CHECK-NEXT: memref.load
//     CHECK-NEXT: secret.yield
//   CHECK-NEXT: }
//   CHECK-NEXT:  secret.cast
//   CHECK-NEXT:  secret.cast
//
//                The main computation
//   CHECK-NEXT:  secret.generic
//   CHECK-NEXT:  ^bb{{.*}}(%[[arg2:.*]]: memref<8xi1>, %[[arg3:.*]]: memref<8xi1>):
//                  Note bit 7 is never loaded because it is shifted out
//   CHECK-DAG:    %[[arg2bit0:.*]] = memref.load %[[arg2]][%[[c0]]] : memref<8xi1>
//   CHECK-DAG:    %[[arg2bit1:.*]] = memref.load %[[arg2]][%[[c1]]] : memref<8xi1>
//   CHECK-DAG:    %[[arg2bit2:.*]] = memref.load %[[arg2]][%[[c2]]] : memref<8xi1>
//   CHECK-DAG:    %[[arg2bit3:.*]] = memref.load %[[arg2]][%[[c3]]] : memref<8xi1>
//   CHECK-DAG:    %[[arg2bit4:.*]] = memref.load %[[arg2]][%[[c4]]] : memref<8xi1>
//   CHECK-DAG:    %[[arg2bit5:.*]] = memref.load %[[arg2]][%[[c5]]] : memref<8xi1>
//   CHECK-DAG:    %[[arg2bit6:.*]] = memref.load %[[arg2]][%[[c6]]] : memref<8xi1>
//
//   CHECK-DAG:    %[[arg3bit0:.*]] = memref.load %[[arg3]][%[[c0]]] : memref<8xi1>
//   CHECK-DAG:    %[[arg3bit1:.*]] = memref.load %[[arg3]][%[[c1]]] : memref<8xi1>
//   CHECK-DAG:    %[[arg3bit2:.*]] = memref.load %[[arg3]][%[[c2]]] : memref<8xi1>
//   CHECK-DAG:    %[[arg3bit3:.*]] = memref.load %[[arg3]][%[[c3]]] : memref<8xi1>
//   CHECK-DAG:    %[[arg3bit4:.*]] = memref.load %[[arg3]][%[[c4]]] : memref<8xi1>
//   CHECK-DAG:    %[[arg3bit5:.*]] = memref.load %[[arg3]][%[[c5]]] : memref<8xi1>
//   CHECK-DAG:    %[[arg3bit6:.*]] = memref.load %[[arg3]][%[[c6]]] : memref<8xi1>
//
//   The order of use of the two allocs seem arbitrary and nondeterministic,
//   so check the stores without the memref names
//   CHECK-DAG:    %[[alloc:.*]] = memref.alloc() : memref<16xi1>
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
//   CHECK-DAG-COUNT-16: memref.store %[[false]], %[[alloc]]
//
//   CHECK-NEXT:    secret.yield {{.*}}, {{.*}}
//   CHECK-NEXT:  }
//   CHECK-NEXT:  secret.cast
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
//   CHECK-NEXT: return
