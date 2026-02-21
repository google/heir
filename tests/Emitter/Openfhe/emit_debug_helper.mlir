// RUN: heir-translate %s --emit-openfhe-pke-debug  --openfhe-debug-helper-include-path=tests/Examples/openfhe/bfv/debug_helper.h | FileCheck %s

!cc = !openfhe.crypto_context
!ct = !openfhe.ciphertext
!pk = !openfhe.public_key
!pt = !openfhe.plaintext
!sk = !openfhe.private_key

// CHECK: include "tests/Examples/openfhe/bfv/debug_helper.h"
// CHECK: void __heir_debug(
// CHECK-SAME: CryptoContextT [[cc:[^,]+]], PrivateKeyT [[sk:[^,]+]], CiphertextT [[ct:[^,]+]],
// CHECK-SAME: const std::map<std::string, std::string>& [[m:debugAttrMap]]) {
// CHECK: auto [[isBlockArgument:.*]] = [[m]].at("asm.is_block_arg");
// CHECK: if ([[isBlockArgument]] == "1")
// CHECK: std::cout << "Input" << std::endl;
// CHECK: else {
// CHECK:   std::cout << [[m]].at("asm.op_name") << std::endl;
// CHECK: }
// CHECK: PlaintextT [[ptxt:.*]];
// CHECK: [[cc]]->Decrypt([[sk]], [[ct]], &[[ptxt]]);
// CHECK: [[ptxt]]->SetLength(std::stod([[m]].at("message.size")));
// CHECK: std::cout << "  " << [[ptxt]] << std::endl;
// CHECK: }

module attributes {scheme.bgv} {
  func.func private @__heir_debug_0(!cc, !sk, !ct)
  func.func @add(%cc: !cc, %sk: !sk, %ct: !ct) -> !ct {
    call @__heir_debug_0(%cc, %sk, %ct) {bound = 50 : i32, random = 3 : i32, complex = {test = 1.2 : f64}, secret.secret} : (!cc, !sk, !ct) -> ()
    return %ct : !ct
  }
}