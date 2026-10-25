// RUN: not heir-translate %s --emit-poulpy 2>&1 | FileCheck %s

!module = !poulpy.module
!scratch = !poulpy.scratch
!ct = memref<!poulpy.ciphertext>
!pt = memref<!poulpy.plaintext>
!sk = !poulpy.secret_key

// encrypt allocates its result itself -- it produces a brand new value, there
// is nothing to "reuse" the way _into ops write into a caller-supplied buffer.
// Passing an ordinary function argument as the ciphertext must be rejected
// rather than silently shadowing it.
func.func @ct_is_argument(%m: !module, %s: !scratch, %ct: !ct, %pt: !pt, %sk: !sk) {
  // CHECK: result must come from an unmaterialized memref.alloc
  poulpy.encrypt %m, %ct, %pt, %sk, %s <base2k = 52, ctk = 300> : (!module, !ct, !pt, !sk, !scratch) -> ()
  return
}
