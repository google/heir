// RUN: heir-opt --secretize --wrap-generic --convert-to-data-oblivious %s | FileCheck %s

// CHECK: test
func.func @test(%secretUpperBound : index{secret.secret}, %secretIndex : index {secret.secret}, %secretTensor : tensor<32xi16>{secret.secret}) -> i16{
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %c0 = arith.constant 0 : i16
    // CHECK: affine.for
    // CHECK-NEXT: arith.cmpi eq
    // CHECK-NEXT: tensor.extract
    // CHECK-NEXT: arith.select
    %extracted = tensor.extract %secretTensor[%secretIndex] : tensor<32xi16>
    // CHECK: affine.for
    // CHECK-NEXT: arith.cmpi slt
    // CHECK-NOT: scf.if
    %result = scf.for %i = %i0 to %secretIndex step %i1 iter_args(%sum = %c0) -> i16 {
        %element = tensor.extract %secretTensor[%i] : tensor<32xi16>
        %cond = arith.cmpi eq, %element, %extracted : i16
        %if = scf.if %cond -> i16 {
            %c2 = arith.constant 2 : i16
            %mul = arith.muli %element, %c2 : i16
            scf.yield %mul : i16
        } else {
            %add = arith.addi %element, %extracted : i16
            scf.yield %add : i16
        }
        %for = arith.addi %sum, %if : i16
        scf.yield %for : i16
    }{lower = 0, upper = 32}
    return %result : i16

}
