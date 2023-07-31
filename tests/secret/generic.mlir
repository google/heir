// RUN: heir-opt %s > %t
// RUN: FileCheck %s < %t

module {
  func.func @main(%value : i32) {
    %X = arith.constant 7 : i32
    %Y = secret.conceal %value : i32 -> !secret.secret<i32>
    // CHECK: secret.generic
    %Z = secret.generic {library_name = "foo"}
      ins(%X, %Y : i32, !secret.secret<i32>) {
      ^bb0(%x: i32, %y: i32) :
        %d = arith.addi %x, %y: i32
        secret.yield %d : i32
      } -> (!secret.secret<i32>)
    func.return
  }
}
