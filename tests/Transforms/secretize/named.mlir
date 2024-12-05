// RUN: heir-opt -secretize=function=comb %s | FileCheck %s

module {
    // CHECK: func.func @comb(%arg0: i1 {secret.secret}, %arg1: i1 {secret.secret})
    func.func @comb(%a: i1, %b: i1) -> () {
        %0 = comb.truth_table %a, %b -> 6 : ui4
        return
    }

    // CHECK: func.func @foo(%arg0: i1, %arg1: i1)
    func.func @foo(%a: i1, %b: i1) -> () {
        %0 = comb.truth_table %a, %b -> 6 : ui4
        return
    }
}
