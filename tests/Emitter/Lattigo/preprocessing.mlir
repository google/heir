// RUN: heir-translate %s --emit-lattigo-preprocessing --package-name=main_utils | FileCheck %s --check-prefix=CHECK-PRE
// RUN: heir-translate %s --emit-lattigo-preprocessed --package-name=main --extra-imports=main_utils | FileCheck %s --check-prefix=CHECK-POST

!pt = !lattigo.rlwe.plaintext
!params = !lattigo.bgv.parameter
!encoder = !lattigo.bgv.encoder

#paramsLiteral = #lattigo.bgv.parameters_literal<
    logN = 14,
    logQ = [56, 55, 55],
    logP = [55],
    plaintextModulus = 0x3ee0001
>

module attributes {scheme.bgv} {
  // CHECK-PRE: package main_utils
  // CHECK-PRE: func Preprocess
  // CHECK-PRE-NOT: func Main

  // CHECK-POST: package main
  // CHECK-POST: import (
  // CHECK-POST:     "main_utils"
  // CHECK-POST: )
  // CHECK-POST: func main
  // CHECK-POST: [[res:.*]] := main_utils.Preprocess
  // CHECK-POST-NOT: func preprocess

  func.func @preprocess(%params: !params, %encoder : !encoder, %value : tensor<4xi32>) -> !pt attributes {client.pack_func = {func_name = "main"}} {
    %pt = lattigo.bgv.new_plaintext %params : (!params) -> !pt
    %res = lattigo.bgv.encode %encoder, %value, %pt {scale = 0} : (!encoder, tensor<4xi32>, !pt) -> !pt
    return %res : !pt
  }

  func.func @main(%params: !params, %encoder : !encoder, %value : tensor<4xi32>) {
    %res = func.call @preprocess(%params, %encoder, %value) : (!params, !encoder, tensor<4xi32>) -> !pt
    return
  }
}
