// RUN: heir-opt -secretize %s | FileCheck %s --check-prefix=CHECK-MULTIPLE --check-prefix=CHECK
// RUN: heir-opt -secretize=function=main %s | FileCheck %s --check-prefix=CHECK-SINGLE --check-prefix=CHECK

module {
    // CHECK-MULTIPLE: func.func @inner(%arg0: i1 {secret.secret}, %arg1: i1 {secret.secret})
    // CHECK-SINGLE: func.func @inner(%arg0: i1, %arg1: i1)
    func.func @inner(%a: i1, %b: i1) -> () {
        %0 = comb.truth_table %a, %b -> 6 : ui4
        return
    }

    // CHECK: func.func @main(%arg0: i1 {secret.secret}, %arg1: i1 {secret.secret})
    func.func @main(%a: i1, %b: i1) -> () {
        func.call @inner(%a, %b) : (i1, i1) -> ()
        return
    }
}
