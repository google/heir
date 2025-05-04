// RUN: heir-opt --shape-inference  %s | FileCheck %s

//CHECK: func @ignore_redundant_attribute
//CHECK-SAME:(%[[X:.*]]: tensor<42xi16> {shape.shape = [111]}, %[[Y:.*]]: tensor<42xi16> {shape.shape = [111]}) -> tensor<42xi16>
func.func @ignore_redundant_attribute(%x: tensor<42xi16> {shape.shape=[111]} , %y: tensor<42xi16> {shape.shape=[111]}) -> (tensor<42xi16>) {
    //CHECK-NEXT: %[[ADD:.*]] = arith.addi %[[X]], %[[Y]] : tensor<42xi16>
    %0 = arith.addi %x, %y : tensor<42xi16>
    //CHECK-NEXT: return %[[ADD]] : tensor<42xi16>
    func.return %0 : tensor<42xi16>
}

//CHECK: func @one_dimensional
//CHECK-SAME:(%[[X:.*]]: tensor<42xi16>, %[[Y:.*]]: tensor<42xi16>) -> tensor<42xi16>
func.func @one_dimensional(%x: tensor<?xi16> {shape.shape=[42]} , %y: tensor<?xi16> {shape.shape=[42]}) -> (tensor<?xi16>) {
    //CHECK-NEXT: %[[ADD:.*]] = arith.addi %[[X]], %[[Y]] : tensor<42xi16>
    %0 = arith.addi %x, %y : tensor<?xi16>
    //CHECK-NEXT: return %[[ADD]] : tensor<42xi16>
    func.return %0 : tensor<?xi16>
}

//CHECK: func @two_dimensional
//CHECK-SAME:(%[[X:.*]]: tensor<2x3xi16>, %[[Y:.*]]: tensor<2x3xi16>) -> tensor<2x3xi16>
func.func @two_dimensional(%x: tensor<?x?xi16> {shape.shape=[2, 3]} , %y: tensor<?x?xi16> {shape.shape=[2, 3]}) -> (tensor<?x?xi16>) {
    //CHECK-NEXT: %[[ADD:.*]] = arith.addi %[[X]], %[[Y]] : tensor<2x3xi16>
    %0 = arith.addi %x, %y : tensor<?x?xi16>
    //CHECK-NEXT: return %[[ADD]] : tensor<2x3xi16>
    func.return %0 : tensor<?x?xi16>
}

//CHECK: func private @decl_only
//CHECK-SAME:(tensor<42xi16>, tensor<42xi16>) -> tensor<42xi16>
func.func private @decl_only(%x: tensor<?xi16> {shape.shape=[42]} , %y: tensor<?xi16> {shape.shape=[42]}) -> (tensor<?xi16> {shape.shape=[42]})

//CHECK: func @affine_for
//CHECK-SAME:(%[[X:.*]]: tensor<42xi16>, %[[Y:.*]]: tensor<42xi16>) -> tensor<42xi16>
func.func @affine_for(%x: tensor<?xi16> {shape.shape=[42]} , %y: tensor<?xi16> {shape.shape=[42]}) -> (tensor<?xi16>) {
    //CHECK-NEXT: %[[FOR:.*]] = affine.for %[[I:.*]] = 0 to 10 iter_args(%[[SUM:.*]] = %[[X]]) -> (tensor<42xi16>) {
    %0 = affine.for %i= 0 to 10 iter_args(%sum = %x) -> (tensor<?xi16>) {
        //CHECK-NEXT: %[[NEW_SUM:.*]] = arith.addi %[[SUM]], %[[Y]] : tensor<42xi16>
        %new_sum = arith.addi %sum, %y : tensor<?xi16>
        //CHECK-NEXT: affine.yield %[[NEW_SUM]] : tensor<42xi16>
        affine.yield %new_sum : tensor<?xi16>
    }
    //CHECK: return %[[FOR]] : tensor<42xi16>
    func.return %0 : tensor<?xi16>
}

//CHECK: func @scf_if
//CHECK-SAME:(%[[X:.*]]: tensor<42xi16>, %[[Y:.*]]: tensor<42xi16>, %[[COND:.*]]: i1) -> tensor<42xi16>
func.func @scf_if(%x: tensor<?xi16> {shape.shape=[42]} , %y: tensor<?xi16> {shape.shape=[42]}, %cond : i1) -> (tensor<?xi16>) {
    //CHECK-NEXT: %[[IF:.*]] = scf.if %[[COND:.*]] -> (tensor<42xi16>) {
    %0 = scf.if %cond -> (tensor<?xi16>) {
        //CHECK-NEXT: %[[ADD:.*]] = arith.addi %[[X]], %[[Y]] : tensor<42xi16>
        %add = arith.addi %x, %y : tensor<?xi16>
        //CHECK-NEXT: scf.yield %[[ADD]] : tensor<42xi16>
        scf.yield %add : tensor<?xi16>
    //CHECK-NEXT: } else {
    } else {
        //CHECK-NEXT: scf.yield %[[X]] : tensor<42xi16>
        scf.yield %x : tensor<?xi16>
    }
    //CHECK: return %[[IF]] : tensor<42xi16>
    func.return %0 : tensor<?xi16>
}

//CHECK: func @scf_while
//CHECK-SAME:(%[[X:.*]]: tensor<42xi16>, %[[Y:.*]]: tensor<42xi16>, %[[COND:.*]]: i1) -> tensor<42xi16>
func.func @scf_while(%x: tensor<?xi16> {shape.shape=[42]} , %y: tensor<?xi16> {shape.shape=[42]}, %cond : i1) -> (tensor<?xi16>) {
    %0 = scf.while (%arg1 = %x) : (tensor<?xi16>) -> tensor<?xi16> {
        %add = arith.addi %arg1, %y : tensor<?xi16>
        scf.condition(%cond) %add : tensor<?xi16>
    } do {
        ^bb0(%arg2: tensor<?xi16>):
            %next = arith.addi %arg2, %y : tensor<?xi16>
            scf.yield %next : tensor<?xi16>
    }
    func.return %0 : tensor<?xi16>
}

// TODO (#1784): Add tests for linalg.generic and similar region carrying non-control-flow ops
