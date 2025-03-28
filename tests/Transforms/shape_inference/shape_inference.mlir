// RUN: heir-opt --shape-inference  %s | FileCheck %s

//CHECK-LABEL: func @one_dimensional
//CHECK-SAME:(%[[X:.*]]: tensor<42xi16>, %[[Y:.*]]: tensor<42xi16>) -> tensor<42xi16>
func.func @one_dimensional(%x: tensor<?xi16> {shape.shape=[42]} , %y: tensor<?xi16> {shape.shape=[42]}) -> (tensor<?xi16>) {
    //CHECK-NEXT: %[[ADD:.*]] = arith.addi %[[X]], %[[Y]] : tensor<42xi16>
    %0 = arith.addi %x, %y : tensor<?xi16>
    //CHECK-NEXT: return %[[ADD]] : tensor<42xi16>
    func.return %0 : tensor<?xi16>
}

//CHECK-LABEL: func @two_dimensional
//CHECK-SAME:(%[[X:.*]]: tensor<2x3xi16>, %[[Y:.*]]: tensor<2x3xi16>) -> tensor<2x3xi16>
func.func @two_dimensional(%x: tensor<?x?xi16> {shape.shape=[2, 3]} , %y: tensor<?x?xi16> {shape.shape=[2, 3]}) -> (tensor<?x?xi16>) {
    //CHECK-NEXT: %[[ADD:.*]] = arith.addi %[[X]], %[[Y]] : tensor<2x3xi16>
    %0 = arith.addi %x, %y : tensor<?x?xi16>
    //CHECK-NEXT: return %[[ADD]] : tensor<2x3xi16>
    func.return %0 : tensor<?x?xi16>
}
