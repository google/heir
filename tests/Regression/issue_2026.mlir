// RUN: heir-opt --convert-secret-for-to-static-for=convert-all-scf-for=true %s | FileCheck %s

module {
  func.func @ctr_mode(%arg0: tensor<16xi8> {secret.secret}, %arg1: tensor<16xi8> {secret.secret}, %arg2: tensor<?x16xi8> {secret.secret}, %arg3: tensor<11x16xi8> {secret.secret}) -> tensor<?x16xi8> {
    %c0_i8 = arith.constant 0 : i8
    %c-1_i8 = arith.constant -1 : i8
    %c16 = arith.constant 16 : index
    %c15 = arith.constant 15 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1_i8 = arith.constant 1 : i8
    %true = arith.constant true
    %false = arith.constant false
    %dim = tensor.dim %arg2, %c0 : tensor<?x16xi8>
    // CHECK-NOT: scf.for
    // CHECK: affine.for
    %0:2 = scf.for %arg4 = %c0 to %dim step %c1 iter_args(%arg5 = %arg2, %arg6 = %arg1) -> (tensor<?x16xi8>, tensor<16xi8>) {
      %1 = func.call @encrypt_block(%arg6, %arg3) : (tensor<16xi8>, tensor<11x16xi8>) -> tensor<16xi8>
      %inserted_slice = tensor.insert_slice %1 into %arg5[%arg4, 0] [1, 16] [1, 1] : tensor<16xi8> into tensor<?x16xi8>
      // CHECK-NOT: scf.for
      // CHECK: affine.for
      %2:2 = scf.for %arg7 = %c0 to %c16 step %c1 iter_args(%arg8 = %arg6, %arg9 = %true) -> (tensor<16xi8>, i1) {
        %3 = arith.subi %c15, %arg7 : index
        %extracted = tensor.extract %arg8[%3] : tensor<16xi8>
        %4 = arith.cmpi eq, %extracted, %c-1_i8 : i8
        %5 = arith.select %arg9, %4, %false : i1
        %6 = scf.if %arg9 -> (tensor<16xi8>) {
          %7 = scf.if %4 -> (tensor<16xi8>) {
            %inserted = tensor.insert %c0_i8 into %arg8[%3] : tensor<16xi8>
            scf.yield %inserted : tensor<16xi8>
          } else {
            %8 = arith.addi %extracted, %c1_i8 : i8
            %inserted = tensor.insert %8 into %arg8[%3] : tensor<16xi8>
            scf.yield %inserted : tensor<16xi8>
          }
          scf.yield %7 : tensor<16xi8>
        } else {
          scf.yield %arg8 : tensor<16xi8>
        }
        scf.yield %6, %5 : tensor<16xi8>, i1
      } {lower = 16 : i64, upper = 16 : i64}
      scf.yield %inserted_slice, %2#0 : tensor<?x16xi8>, tensor<16xi8>
    } {lower = 0 : i64, upper = 64 : i64}
    return %0#0 : tensor<?x16xi8>
  }
  func.func @encrypt_block(%arg0: tensor<16xi8> {secret.secret}, %arg1: tensor<11x16xi8> {secret.secret}) -> tensor<16xi8> {
    %extracted_slice = tensor.extract_slice %arg1[0, 0] [1, 16] [1, 1] : tensor<11x16xi8> to tensor<16xi8>
    %0 = arith.xori %arg0, %extracted_slice : tensor<16xi8>
    return %0 : tensor<16xi8>
  }
}
