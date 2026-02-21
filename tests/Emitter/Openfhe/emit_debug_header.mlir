// RUN: heir-translate %s --emit-openfhe-pke-debug-header | FileCheck %s

!cc = !openfhe.crypto_context
!ct = !openfhe.ciphertext
!pk = !openfhe.public_key
!pt = !openfhe.plaintext
!sk = !openfhe.private_key


module attributes {scheme.bgv} {
  // CHECK: #include <map>
  // CHECK: #include <string>
  // CHECK: #include "openfhe/pke/openfhe.h" // from @openfhe
  // CHECK: void __heir_debug(
  // CHECK-NOT: __heir_debug_0
  // CHECK-SAME: CryptoContextT [[cc:[^,]+]], PrivateKeyT [[sk:[^,]+]], CiphertextT [[ct:[^,]+]],
  // CHECK-SAME: const std::map<std::string, std::string>& [[m:debugAttrMap]]
  // CHECK-SAME: );
  func.func private @__heir_debug_0(!cc, !sk, !ct)
  func.func @add(%cc: !cc, %sk: !sk, %ct: !ct) -> !ct {
    call @__heir_debug_0(%cc, %sk, %ct) {bound = 50 : i32, random = 3 : i32, complex = {test = 1.2 : f64}, secret.secret} : (!cc, !sk, !ct) -> ()
    return %ct : !ct
  }
}
