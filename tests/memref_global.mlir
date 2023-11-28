// This verifies that the memref.global was removed and that its constant values
// are forwarded to referencing affine loads.

// RUN: heir-opt --memref-global-replace %s | FileCheck %s

// The following verifies that the memref values were unchanged after the pass.

// RUN: mlir-opt %s -pass-pipeline="builtin.module( \
// RUN:     affine-expand-index-ops, \
// RUN:     lower-affine, \
// RUN:     finalize-memref-to-llvm, \
// RUN:     func.func( \
// RUN:       convert-scf-to-cf, \
// RUN:       affine-expand-index-ops, \
// RUN:       convert-arith-to-llvm, \
// RUN:       lower-affine), \
// RUN:     convert-func-to-llvm, \
// RUN:     convert-cf-to-llvm, \
// RUN:     reconcile-unrealized-casts)" | \
// RUN:   mlir-cpu-runner -e main -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_runner_utils" | \
// RUN:   FileCheck %s --check-prefix CHECK_PREPASS --allow-empty

// RUN: heir-opt --memref-global-replace %s | \
// RUN:   mlir-opt -pass-pipeline="builtin.module( \
// RUN:     affine-expand-index-ops, \
// RUN:     lower-affine, \
// RUN:     finalize-memref-to-llvm, \
// RUN:     func.func( \
// RUN:       convert-scf-to-cf, \
// RUN:       affine-expand-index-ops, \
// RUN:       convert-arith-to-llvm, \
// RUN:       lower-affine), \
// RUN:     convert-func-to-llvm, \
// RUN:     convert-cf-to-llvm, \
// RUN:     reconcile-unrealized-casts)" | \
// RUN:   mlir-cpu-runner -e main -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_runner_utils" | \
// RUN:   FileCheck %s --check-prefix CHECK_POSTPASS --allow-empty

// CHECK_PREPASS-NOT: MISMATCH
// CHECK_POSTPASS-NOT: MISMATCH

// CHECK-LABEL: module
module {
  // CHECK-NOT: memref.global
  memref.global "private" constant @__constant_8xi16 : memref<8xi16> = dense<[-10, 20, 3, 4, 5, 6, 7, 8]>

  // Test helpers
  llvm.mlir.global internal constant @str_fail("MISMATCH\0A")
  func.func private @printString(!llvm.ptr) -> ()
  // Prints 'MISMATCH' to stdout.
  func.func @printMismatch() -> () {
    %0 = llvm.mlir.addressof @str_fail : !llvm.ptr
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.getelementptr %0[%1, %1]
      : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<9 x i8>
    func.call @printString(%2) : (!llvm.ptr) -> ()
    return
  }
  func.func @check_int(%lhs: i16, %rhs: i16) -> () {
    %mismatch = arith.cmpi ne, %lhs, %rhs : i16
    scf.if %mismatch -> () {
      func.call @printMismatch() : () -> ()
    }
    return
  }

  // CHECK-LABEL: func @main
  func.func @main() -> () {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    // CHECK-NOT: memref.get_global
    %0 = memref.get_global @__constant_8xi16 : memref<8xi16>

    // CHECK-NOT affine.load
    %1 = affine.load %0[%c0] :  memref<8xi16>
    // CHECK: arith.constant -10
    %c_-10 = arith.constant -10 : i16
    func.call @check_int(%1, %c_-10) : (i16, i16) -> ()

    // CHECK-NOT affine.load
    %2 = affine.load %0[%c1] :  memref<8xi16>
    // CHECK: arith.constant 20
    %c_20 = arith.constant 20 : i16
    func.call @check_int(%2, %c_20) : (i16, i16) -> ()

    // CHECK-NOT affine.load
    %3 = affine.load %0[%c2] :  memref<8xi16>
    // CHECK: arith.constant 3
    %c_3 = arith.constant 3 : i16
    func.call @check_int(%3, %c_3) : (i16, i16) -> ()

    // CHECK-NOT affine.load
    %4 = affine.load %0[%c3] :  memref<8xi16>
    // CHECK: arith.constant 4
    %c_4 = arith.constant 4 : i16
    func.call @check_int(%4, %c_4) : (i16, i16) -> ()

    // CHECK-NOT affine.load
    %5 = affine.load %0[%c4] :  memref<8xi16>
    // CHECK: arith.constant 5
    %c_5 = arith.constant 5 : i16
    func.call @check_int(%5, %c_5) : (i16, i16) -> ()

    // CHECK-NOT affine.load
    %6 = affine.load %0[%c5] :  memref<8xi16>
    // CHECK: arith.constant 6
    %c_6 = arith.constant 6 : i16
    func.call @check_int(%6, %c_6) : (i16, i16) -> ()

    // CHECK-NOT affine.load
    %7 = affine.load %0[%c6] :  memref<8xi16>
    // CHECK: arith.constant 7
    %c_7 = arith.constant 7 : i16
    func.call @check_int(%7, %c_7) : (i16, i16) -> ()

    // CHECK-NOT affine.load
    %8 = affine.load %0[%c7] :  memref<8xi16>
    // CHECK: arith.constant 8
    %c_8 = arith.constant 8 : i16
    func.call @check_int(%8, %c_8) : (i16, i16) -> ()

    // CHECK: return
    return
  }
}
