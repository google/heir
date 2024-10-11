// The following verifies that the memref.global was removed and that its constant values
// are forwarded to referencing affine loads when the global is represented with
// raw values.

// RUN: heir-opt --memref-global-replace %s | FileCheck %s

// The following validates correctness of the model before and after the
// memref-global-replace pass.

// RUN: heir-opt %s -pass-pipeline="builtin.module( \
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
// RUN:   heir-opt -pass-pipeline="builtin.module( \
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
  memref.global "private" constant @__constant_4x4xi16 : memref<4x4xi16> = dense<"0xFFF9FD0A0708070307F2D109F0E92809DF05FAF0E8E3130E08EFD3EE0FE8EB14">

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

  func.func @main() -> () {
    // CHECK-NOT: memref.get_global
    %global = memref.get_global @__constant_4x4xi16 : memref<4x4xi16>
    %c_0 = arith.constant 0 : index
    %c_1 = arith.constant 1 : index
    %c_2 = arith.constant 2 : index
    %c_3 = arith.constant 3 : index

    // CHECK: arith.constant -1537
    %1 = affine.load %global[%c_0, %c_0] :  memref<4x4xi16>
    %c_-1537 = arith.constant -1537 : i16
    func.call @check_int(%1, %c_-1537) : (i16, i16) -> ()

    // CHECK: arith.constant -5648
    %2 = affine.load %global[%c_1, %c_2] :  memref<4x4xi16>
    %c_-5648 = arith.constant -5648 : i16
    func.call @check_int(%2, %c_-5648) : (i16, i16) -> ()

    // CHECK: arith.constant 5355
    %3 = affine.load %global[%c_3, %c_3] :  memref<4x4xi16>
    %c_5355 = arith.constant 5355 : i16
    func.call @check_int(%3, %c_5355) : (i16, i16) -> ()

    // CHECK: return
    return
  }
}
