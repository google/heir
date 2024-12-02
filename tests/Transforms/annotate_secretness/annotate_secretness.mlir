// RUN: heir-opt --annotate-secretness  %s | FileCheck %s

// CHECK-LABEL: @annotated_secretness
// CHECK-SAME:([[S:%.*]]: [[T:.*]] {secret.secret}, [[P:%.*]]: [[T]])
func.func @annotated_secretness(%s: i32 {secret.secret}, %p: i32) {
    //CHECK-NEXT: arith.addi  [[P]], [[P]] {secretness = false} : [[T]]
    %0 = arith.addi %p, %p : i32
    //CHECK-NEXT: arith.addi  [[S]], [[P]] {secretness = true} : [[T]]
    %1 = arith.addi %s, %p : i32
    func.return
}

// CHECK-LABEL: @typed_secretness
// CHECK-SAME: ([[S:%.*]]: [[ST:.*]], [[P:%.*]]: [[PT:.*]])
func.func @typed_secretness(%s: !secret.secret<i32>, %p: i32) {
    %0 = secret.generic ins(%s , %p : !secret.secret<i32>, i32) {
    //CHECK: ^bb0([[SS:%.*]]: [[PT]], [[PP:%.*]]: [[PT]]):
    ^bb0(%ss: i32, %pp: i32):
        //CHECK-NEXT: arith.addi  [[PP]], [[PP]] {secretness = false} : [[PT]]
        %0 = arith.addi %pp, %pp : i32
        //CHECK-NEXT: arith.addi  [[SS]], [[PP]] {secretness = true} : [[PT]]
        %1 = arith.addi %ss, %pp : i32
        secret.yield  %1 : i32
    } -> (!secret.secret<i32>)
    func.return
}

// CHECK-LABEL: @multi_result_secretness
// CHECK-SAME: ([[S:%.*]]: [[T:.*]] {secret.secret}, [[P:%.*]]: [[T]])
func.func @multi_result_secretness(%s: i32 {secret.secret}, %p: i32) {
    //CHECK-NEXT: arith.addui_extended [[P]], [[P]] {result_0_secretness = false, result_1_secretness = false} : [[T]], i1
    %p1, %p2 = arith.addui_extended %p, %p : i32, i1
    //CHECK-NEXT: arith.addui_extended [[S]], [[P]] {result_0_secretness = true, result_1_secretness = true} : [[T]], i1
    %s1, %s2 = arith.addui_extended %s, %p : i32, i1
    func.return
}
