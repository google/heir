// RUN: heir-opt --yosys-optimizer %s

module {
  func.func @submod(%arg0: i8) -> (i8) {
    %c1 = arith.constant 1 : i8
    %0 = arith.addi %arg0, %c1 : i8
    return %0 : i8
  }
  // CHECK-LABEL: @main
  // CHECK: secret.generic
  // CHECK-NOT: submod
  // CHECK: secret.yield
  // CHECK: return
  func.func @main(%arg0: !secret.secret<i8>) -> (!secret.secret<i8>) {
    %0 = secret.generic ins(%arg0 : !secret.secret<i8>) {
      ^bb0(%ARG0: i8) :
        %1 = func.call @submod(%ARG0) : (i8) -> i8
        secret.yield %1 : i8
    } -> !secret.secret<i8>
    return %0 : !secret.secret<i8>
  }
}
