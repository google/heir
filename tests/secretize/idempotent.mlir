// RUN: heir-opt -secretize -wrap-generic -secretize -wrap-generic %s | FileCheck %s

// This is a regression test that ensures that this sequence of patterns doesn't
// double-wrap types a secret.generic block. One fix was to ensure that
// secretize does not double wrap secret types.

module {
    // CHECK: @main(%[[arg:.*]]: !secret.secret<i1>)
    // CHECK-COUNT-1: secret.generic
    // CHECK-NOT: secret.generic
    // CHECK: return
    func.func @main(%a: i1) -> i1 {
        %0 = comb.truth_table %a, %a -> 6 : ui4
        return %0 : i1
    }
}
