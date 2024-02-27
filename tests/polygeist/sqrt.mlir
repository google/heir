// RUN: heir-opt --secretize=entry-function=_Z5isqrti --wrap-generic --secret-distribute-generic %s


module {
  // Note: this should be i16, but polygeist failed to emit valid MLIR when the
  // input is a short instead of an int see
  // https://github.com/llvm/Polygeist/issues/387
  //
  // Also note that this could not be raised to affine because it is naturally
  // iteration-dependent. This makes it a bad candidate for SIMD-style FHE,
  // so we're mainly using it to exercise the compilation chain.
  //
  // The secret annotation is added manually, not by polygeist.
  func.func @_Z5isqrti(%arg0: i32) -> i32 {
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16384_i32 = arith.constant 16384 : i32
    %c0_i32 = arith.constant 0 : i32
    %0:3 = affine.for %arg1 = 0 to 8 iter_args(%arg2 = %c16384_i32, %arg3 = %c0_i32, %arg4 = %arg0) -> (i32, i32, i32) {
      %1 = arith.addi %arg3, %arg2 : i32
      %2 = arith.cmpi sge, %arg4, %1 : i32
      %3:2 = scf.if %2 -> (i32, i32) {
        %5 = arith.subi %arg4, %1 : i32
        %6 = arith.shrsi %arg3, %c1_i32 : i32
        %7 = arith.addi %6, %arg2 : i32
        scf.yield %7, %5 : i32, i32
      } else {
        %5 = arith.shrsi %arg3, %c1_i32 : i32
        scf.yield %5, %arg4 : i32, i32
      }
      %4 = arith.shrsi %arg2, %c2_i32 : i32
      affine.yield %4, %3#0, %3#1 : i32, i32, i32
    }
    return %0#1 : i32
  }
}
