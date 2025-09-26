// RUN: heir-opt %s --yosys-optimizer --secret-distribute-generic --canonicalize --secret-to-cggi | FileCheck %s

// CHECK-NOT: secret
func.func @sum(%arr: !secret.secret<tensor<8xi8>>) -> !secret.secret<i8> {
    %c0 = arith.constant 0 : i8
    %s0 = secret.conceal %c0 : i8 -> !secret.secret<i8>
    %retval = affine.for %i = 0 to 8 step 1 iter_args(%acc = %s0) -> (!secret.secret<i8>) {
        %new_acc = secret.generic(%arr: !secret.secret<tensor<8xi8>>, %acc: !secret.secret<i8>) {
            ^bb0(%ARR: tensor<8xi8>, %ACC: i8):
                %cur = tensor.extract %ARR[%i] : tensor<8xi8>
                %sum = arith.addi %cur, %ACC : i8
                secret.yield %sum : i8
        } -> (!secret.secret<i8>)
        affine.yield %new_acc : !secret.secret<i8>
    }
    return %retval : !secret.secret<i8>
}
