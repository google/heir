// RUN: heir-opt --secretize --wrap-generic --cse --canonicalize --secret-insert-mgmt-bgv --generate-param-bgv --secret-distribute-generic --secret-to-bgv=poly-mod-degree=32 --bgv-to-lwe --lwe-to-openfhe --openfhe-configure-crypto-context=entry-function=complex_func %s | FileCheck %s

!ty = tensor<32xi16>

func.func @complex_func(%arg0 : !ty {secret.secret}, %arg1 : !ty {secret.secret}) -> !ty {
  %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]> : tensor<32xi16>
  %0 = arith.muli %arg0, %arg1 : !ty
  %1 = arith.addi %0, %arg0 : !ty
  %2 = arith.muli %0, %cst : !ty

  %4 = arith.muli %arg0, %cst : !ty
  %5 = arith.muli %4, %arg1 : !ty
  %6 = arith.subi %5, %4 : !ty
  %7 = arith.muli %5, %cst : !ty

  %ret = arith.addi %1, %7 : !ty
  return %ret : !ty
}

// CHECK: @complex_func
// CHECK: @complex_func__generate_crypto_context
// CHECK: mulDepth = 3

// CHECK: @complex_func__configure_crypto_context
// CHECK: openfhe.gen_mulkey
