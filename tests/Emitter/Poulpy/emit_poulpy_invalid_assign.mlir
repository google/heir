// RUN: not heir-translate %s --emit-poulpy 2>&1 | FileCheck %s

!module = !poulpy.module<ntt4x30_ref>
!scratch = !poulpy.scratch
!ct = memref<!poulpy.ciphertext>

// An _assign op reads its dst before writing it. A dst that comes straight
// from memref.alloc() with no prior initializing op has no real value to
// read, so this must be rejected rather than silently emitting Rust that
// reads uninitialized memory.
func.func @uninitialized_assign(%m: !module, %s: !scratch, %a: !ct) {
  %dst = memref.alloc() : !ct
  // CHECK: dst has not been initialized before this use
  poulpy.add_assign %m, %dst, %a, %s : (!module, !ct, !ct, !scratch) -> ()
  return
}
