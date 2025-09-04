// RUN: heir-opt --annotate-secretness='verbose=true'  %s

func.func @annotated_secretness(%s: i32 {secret.secret}, %p: i32) {
    %0 = arith.addi %p, %p : i32
    %1 = arith.addi %s, %p : i32
    return
}

func.func @typed_secretness(%s: !secret.secret<i32>, %p: i32) {
    %0 = secret.generic(%s: !secret.secret<i32>, %p: i32) {
    ^body(%ss: i32, %pp: i32):
        %0 = arith.addi %pp, %pp : i32
        %1 = arith.addi %ss, %pp : i32
        secret.yield  %1 : i32
    } -> (!secret.secret<i32>)
    return
}

func.func @multi_result_secretness(%s: i32 {secret.secret}, %p: i32) {
    %p1, %p2 = arith.addui_extended %p, %p : i32, i1
    %s1, %s2 = arith.addui_extended %s, %p : i32, i1
    return
}

func.func @return_secretness(%s: i32 {secret.secret}, %p: i32) -> (i32) {
    %0 = arith.addi %s, %p : i32
    return %0 : i32
}

func.func private @callee(!secret.secret<i32>) -> !secret.secret<i32>
func.func @func_call(%s: !secret.secret<i32>) {
    %1 = func.call @callee(%s) : (!secret.secret<i32>) -> !secret.secret<i32>
    func.return
}
