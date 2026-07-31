// RUN: not heir-translate %s --emit-poulpy 2>&1 | FileCheck %s

!module = !poulpy.module<ntt4x30_ref>
!scratch = !poulpy.scratch
!ct = memref<!poulpy.ciphertext>
!ctu = memref<!poulpy.unnormalized_ciphertext>

// normalize's result must come from an unmaterialized memref.alloc -- it
// produces a brand new value, there is nothing to "reuse" the way _into ops
// write into a caller-supplied buffer. Passing an ordinary function argument
// as res must be rejected rather than silently shadowing it.
func.func @res_is_argument(%m: !module, %s: !scratch, %res: !ct, %a: !ctu) {
  // CHECK: normalize result must come from an unmaterialized memref.alloc
  poulpy.normalize %m, %res, %a, %s : (!module, !ct, !ctu, !scratch) -> ()
  return
}
