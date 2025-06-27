// RUN: heir-opt %s "--mlir-to-bgv=enable-arithmetization=false" | FileCheck %s

// A program that does not require any kind of arithmetization (i.e., already consists purely of operations that we can lower directly to FHE)
// should be able to be lowered to FHE successfully without running the arithmetization stage.

// CHECK: @foo
func.func @foo(%x: tensor<1024xi16> {secret.secret}, %y: tensor<1024xi16> {secret.secret}) -> tensor<1024xi16> {
    // CHECK: bgv.add
    %add = arith.addi %x, %y : tensor<1024xi16>
    // CHECK: bgv.sub
    %sub = arith.subi %x, %y : tensor<1024xi16>
    // CHECK: bgv.mul
    %mul = arith.muli %add, %sub : tensor<1024xi16>
    return %mul : tensor<1024xi16>
}

// CHECK: @foo__encrypt__
// CHECK-NOT: affine.for
// CHECK-NOT: tensor.extract

// CHECK: @foo__decrypt__
// CHECK-NOT: affine.for
// CHECK-NOT: tensor.insert
