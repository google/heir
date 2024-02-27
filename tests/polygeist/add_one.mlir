// RUN: heir-opt --secretize=entry-function=_Z7add_onec --wrap-generic --secret-distribute-generic %s

module {
  func.func @_Z7add_onec(%arg0: i8) -> i8 {
    // Note: polygeist does this unnecessary extsi/trunci because it defines
    // the constant as an i32. We may want to add a pass to narrow this.
    // Note that `arith-int-narrowing` helps:
    //
    //    '--arith-int-narrowing=int-bitwidths-supported=1,2,4,8,16,32' --canonicalize
    //
    // results in the constant 1 being defined as an i16, presumably because
    // adding 1 to an 8-bit integer can cause it to overflow, so it has to
    // guarantee the same results of having an i32 addition truncated, and this
    // probably must be fixed at the polygeist level by having the constant be
    // defined as an i8. See https://github.com/llvm/Polygeist/issues/388
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.extsi %arg0 : i8 to i32
    %1 = arith.addi %0, %c1_i32 : i32
    %2 = arith.trunci %1 : i32 to i8
    return %2 : i8
  }
}
