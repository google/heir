// RUN: heir-translate %s --emit-lattigo-debug | FileCheck %s

!ct = !lattigo.rlwe.ciphertext
!encryptor = !lattigo.rlwe.encryptor<publicKey = true>
!decryptor = !lattigo.rlwe.decryptor
!evaluator = !lattigo.bgv.evaluator
!encoder = !lattigo.bgv.encoder
!params = !lattigo.bgv.parameter


// CHECK: package [[package:.*]]
// CHECK: import (
// CHECK:   "fmt"
// CHECK:   "github.com/tuneinsight/lattigo/v6/core/rlwe"
// CHECK:   "github.com/tuneinsight/lattigo/v6/schemes/bgv"
// CHECK:   "strconv"
// CHECK: )
// CHECK: func __heir_debug(
// CHECK-SAME: [[eval:[^ ]+]] *bgv.Evaluator,
// CHECK-SAME: [[param:[^ ]+]] bgv.Parameters,
// CHECK-SAME: [[encoder:[^ ]+]] *bgv.Encoder,
// CHECK-SAME: [[decryptor:[^ ]+]] *rlwe.Decryptor,
// CHECK-SAME: [[ct:[^ ]+]] *rlwe.Ciphertext,
// CHECK-SAME: [[m:debugAttrMap]] map[string]string) {
// CHECK: isBlockArgument := [[m]]["asm.is_block_arg"]
// CHECK: if isBlockArgument == "1" {
// CHECK:   fmt.Println("Input")
// CHECK: } else {
// CHECK:   fmt.Println([[m]]["asm.op_name"])
// CHECK: }
// CHECK: messageSize, _ := strconv.Atoi([[m]]["message.size"])
// CHECK: value := make([]int64, messageSize)
// CHECK: pt := [[decryptor]].DecryptNew([[ct]])
// CHECK: [[encoder]].Decode(pt, value)
// CHECK: fmt.Printf("  %v\n", value)
// CHECK: }
module attributes {scheme.bgv} {
  func.func private @__heir_debug_0(!lattigo.bgv.evaluator, !lattigo.bgv.parameter, !lattigo.bgv.encoder, !lattigo.rlwe.decryptor, !lattigo.rlwe.ciphertext)
  func.func @dot_product(%evaluator: !lattigo.bgv.evaluator, %param: !lattigo.bgv.parameter, %encoder: !lattigo.bgv.encoder, %decryptor: !lattigo.rlwe.decryptor, %ct: !lattigo.rlwe.ciphertext, %ct_0: !lattigo.rlwe.ciphertext) -> !lattigo.rlwe.ciphertext {
    call @__heir_debug_0(%evaluator, %param, %encoder, %decryptor, %ct) {bound = "50", random = 3, complex = {test = 1.2}, secret.secret} : (!lattigo.bgv.evaluator, !lattigo.bgv.parameter, !lattigo.bgv.encoder, !lattigo.rlwe.decryptor, !lattigo.rlwe.ciphertext) -> ()
    return %ct : !lattigo.rlwe.ciphertext
  }
}
