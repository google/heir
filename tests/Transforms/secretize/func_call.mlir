// RUN: heir-opt -wrap-generic %s | FileCheck %s

module {
    // CHECK: private @some_func(i1) -> i1
    func.func private @some_func(i1) -> i1

    // CHECK: private @some_func2(!secret.secret<i1>) -> !secret.secret<i1>
    func.func private @some_func2(i1 {secret.secret}) -> i1

    // CHECK: @main(%[[arg:.*]]: !secret.secret<i1>)
    // CHECK-COUNT-1: secret.generic
    // CHECK-NOT: secret.generic
    // CHECK: return
    func.func @main(%a: i1 {secret.secret}) -> i1 {
        %0 = func.call @some_func(%a) : (i1) -> i1
        return %0 : i1
    }
}
