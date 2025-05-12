// RUN: heir-translate --emit-verilog --verify-diagnostics %s

module {
  func.func @add_one(%in: !secret.secret<i8>) -> (!secret.secret<i8>) {
    %one = arith.constant 1 : i8
    // expected-error@+1 {{allowSecretOps is false, but encountered a secret op.}}
    %1 = secret.generic
        (%in: !secret.secret<i8>, %one: i8) {
        ^bb0(%IN: i8, %ONE: i8) :
            %2 = arith.addi %IN, %ONE : i8
            secret.yield %2 : i8
        } -> (!secret.secret<i8>)
    return %1 : !secret.secret<i8>
  }
}
