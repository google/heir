// RUN: heir-opt --annotate-secretness='verbose=true'  %s | FileCheck %s

// CHECK-LABEL: @annotated_secretness
// CHECK-SAME:([[S:%.*]]: [[T:.*]] {secret.secret}, [[P:%.*]]: [[T]] {secret.public})
func.func @annotated_secretness(%s: i32 {secret.secret}, %p: i32) {
    //CHECK-NEXT: arith.addi  [[P]], [[P]] {secret.public} : [[T]]
    %0 = arith.addi %p, %p : i32
    //CHECK-NEXT: arith.addi  [[S]], [[P]] {secret.secret} : [[T]]
    %1 = arith.addi %s, %p : i32
    return
}

// CHECK-LABEL: @typed_secretness
// CHECK-SAME: ([[S:%.*]]: [[ST:.*]], [[P:%.*]]: [[PT:.*]] {secret.public})
func.func @typed_secretness(%s: !secret.secret<i32>, %p: i32) {
    %0 = secret.generic ins(%s , %p : !secret.secret<i32>, i32) {
    //CHECK: ^body([[SS:%.*]]: [[PT]], [[PP:%.*]]: [[PT]]):
    ^body(%ss: i32, %pp: i32):
        //CHECK-NEXT: arith.addi  [[PP]], [[PP]] {secret.public} : [[PT]]
        %0 = arith.addi %pp, %pp : i32
        //CHECK-NEXT: arith.addi  [[SS]], [[PP]] {secret.secret} : [[PT]]
        %1 = arith.addi %ss, %pp : i32
        secret.yield  %1 : i32
    } -> (!secret.secret<i32>)
    return
}

// CHECK-LABEL: @multi_result_secretness
// CHECK-SAME: ([[S:%.*]]: [[T:.*]] {secret.secret}, [[P:%.*]]: [[T]] {secret.public})
func.func @multi_result_secretness(%s: i32 {secret.secret}, %p: i32) {
    //CHECK-NEXT: arith.addui_extended [[P]], [[P]] {secretness = [{secret.public}, {secret.public}]} : [[T]], i1
    %p1, %p2 = arith.addui_extended %p, %p : i32, i1
    //CHECK-NEXT: arith.addui_extended [[S]], [[P]] {secretness = [{secret.secret}, {secret.secret}]} : [[T]], i1
    %s1, %s2 = arith.addui_extended %s, %p : i32, i1
    return
}

// CHECK-LABEL: @return_secretness
// CHECK-SAME: ([[S:%.*]]: [[T:.*]] {secret.secret}, [[P:%.*]]: [[T]] {secret.public}) -> ([[T]] {secret.secret})
func.func @return_secretness(%s: i32 {secret.secret}, %p: i32) -> (i32) {
    //CHECK-NEXT: arith.addi  [[S]], [[P]] {secret.secret} : i32
    %0 = arith.addi %s, %p : i32
    //CHECK-NEXT: return {secret.secret} [[R:%.*]] : i32
    return %0 : i32
}
