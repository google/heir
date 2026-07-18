// RUN: heir-opt %s | FileCheck %s

// This simply tests for syntax.

!module    = !poulpy.module<fft64_ref>
!ct        = !poulpy.ciphertext
!unnorm_ct = !poulpy.unnormalized_ciphertext
!pt        = !poulpy.plaintext
!scratch   = !poulpy.scratch
!sk        = !poulpy.secret_key
!tk        = !poulpy.tensor_key
!akm       = !poulpy.automorphism_key_map
!bsk       = !poulpy.bootstrapping_keys
!bctx      = !poulpy.bootstrapping_context

module {
  // CHECK: func @test_module_create
  func.func @test_module_create() {
    // CHECK: poulpy.module_create
    %mod = poulpy.module_create {N = 64 : i64} : () -> !module
    return
  }

  // CHECK: func @test_scratch_alloc
  func.func @test_scratch_alloc() {
    // CHECK: poulpy.scratch_alloc
    %scratch = poulpy.scratch_alloc {size = 1024 : i64} : () -> !scratch
    return
  }

  // CHECK: func @test_types
  // Exercises MemRefElementTypeInterface on ciphertext, unnormalized_ciphertext, and plaintext.
  func.func @test_types(
    %ct_buf: memref<!ct>,
    %unnorm_buf: memref<!unnorm_ct>,
    %pt_buf: memref<!pt>,
    %scratch: !scratch,
    %sk: !sk,
    %tk: !tk,
    %akm: !akm,
    %bsk: !bsk,
    %bctx: !bctx
  ) {
    return
  }

  // CHECK: func @test_encode
  func.func @test_encode(%mod: !module, %pt: memref<!pt>, %re: memref<f64>, %im: memref<f64>) {
    // CHECK: poulpy.encode
    poulpy.encode %mod, %pt, %re, %im {logDelta = 40 : i64, logBudget = 20 : i64} : (!module, memref<!pt>, memref<f64>, memref<f64>) -> ()
    return
  }

  // CHECK: func @test_decode
  func.func @test_decode(%mod: !module, %re: memref<f64>, %im: memref<f64>, %pt: memref<!pt>) {
    // CHECK: poulpy.decode
    poulpy.decode %mod, %re, %im, %pt : (!module, memref<f64>, memref<f64>, memref<!pt>) -> ()
    return
  }

  // CHECK: func @test_encrypt
  func.func @test_encrypt(%mod: !module, %ct: memref<!ct>, %pt: memref<!pt>, %sk: !sk, %scratch: !scratch) {
    // CHECK: poulpy.encrypt
    poulpy.encrypt %mod, %ct, %pt, %sk, %scratch : (!module, memref<!ct>, memref<!pt>, !sk, !scratch) -> ()
    return
  }

  // CHECK: func @test_decrypt
  func.func @test_decrypt(%mod: !module, %pt: memref<!pt>, %ct: memref<!ct>, %sk: !sk, %scratch: !scratch) {
    // CHECK: poulpy.decrypt
    poulpy.decrypt %mod, %pt, %ct, %sk, %scratch : (!module, memref<!pt>, memref<!ct>, !sk, !scratch) -> ()
    return
  }

  // CHECK: func @test_add
  func.func @test_add(%mod: !module, %dst: memref<!ct>, %a: memref<!ct>, %b: memref<!ct>, %scratch: !scratch) {
    // CHECK: poulpy.add
    poulpy.add %mod, %dst, %a, %b, %scratch : (!module, memref<!ct>, memref<!ct>, memref<!ct>, !scratch) -> ()
    return
  }

  // CHECK: func @test_add_assign
  func.func @test_add_assign(%mod: !module, %dst: memref<!ct>, %a: memref<!ct>, %scratch: !scratch) {
    // CHECK: poulpy.add_assign
    poulpy.add_assign %mod, %dst, %a, %scratch : (!module, memref<!ct>, memref<!ct>, !scratch) -> ()
    return
  }

  // CHECK: func @test_add_unnormalized
  func.func @test_add_unnormalized(%mod: !module, %dst: memref<!unnorm_ct>, %a: memref<!ct>, %b: memref<!ct>, %scratch: !scratch) {
    // CHECK: poulpy.add_unnormalized
    poulpy.add_unnormalized %mod, %dst, %a, %b, %scratch : (!module, memref<!unnorm_ct>, memref<!ct>, memref<!ct>, !scratch) -> ()
    return
  }

  // CHECK: func @test_sub
  func.func @test_sub(%mod: !module, %dst: memref<!ct>, %a: memref<!ct>, %b: memref<!ct>, %scratch: !scratch) {
    // CHECK: poulpy.sub
    poulpy.sub %mod, %dst, %a, %b, %scratch : (!module, memref<!ct>, memref<!ct>, memref<!ct>, !scratch) -> ()
    return
  }

  // CHECK: func @test_sub_assign
  func.func @test_sub_assign(%mod: !module, %dst: memref<!ct>, %a: memref<!ct>, %scratch: !scratch) {
    // CHECK: poulpy.sub_assign
    poulpy.sub_assign %mod, %dst, %a, %scratch : (!module, memref<!ct>, memref<!ct>, !scratch) -> ()
    return
  }

  // CHECK: func @test_sub_unnormalized
  func.func @test_sub_unnormalized(%mod: !module, %dst: memref<!unnorm_ct>, %a: memref<!ct>, %b: memref<!ct>, %scratch: !scratch) {
    // CHECK: poulpy.sub_unnormalized
    poulpy.sub_unnormalized %mod, %dst, %a, %b, %scratch : (!module, memref<!unnorm_ct>, memref<!ct>, memref<!ct>, !scratch) -> ()
    return
  }

  // CHECK: func @test_mul
  func.func @test_mul(%mod: !module, %dst: memref<!ct>, %a: memref<!ct>, %b: memref<!ct>, %tk: !tk, %scratch: !scratch) {
    // CHECK: poulpy.mul
    poulpy.mul %mod, %dst, %a, %b, %tk, %scratch : (!module, memref<!ct>, memref<!ct>, memref<!ct>, !tk, !scratch) -> ()
    return
  }

  // CHECK: func @test_mul_assign
  func.func @test_mul_assign(%mod: !module, %dst: memref<!ct>, %a: memref<!ct>, %tk: !tk, %scratch: !scratch) {
    // CHECK: poulpy.mul_assign
    poulpy.mul_assign %mod, %dst, %a, %tk, %scratch : (!module, memref<!ct>, memref<!ct>, !tk, !scratch) -> ()
    return
  }

  // CHECK: func @test_rotate
  func.func @test_rotate(%mod: !module, %dst: memref<!ct>, %src: memref<!ct>, %akm: !akm, %scratch: !scratch) {
    // CHECK: poulpy.rotate
    poulpy.rotate %mod, %dst, %src, %akm, %scratch {k = 1 : i64} : (!module, memref<!ct>, memref<!ct>, !akm, !scratch) -> ()
    return
  }

  // CHECK: func @test_rotate_assign
  func.func @test_rotate_assign(%mod: !module, %dst: memref<!ct>, %akm: !akm, %scratch: !scratch) {
    // CHECK: poulpy.rotate_assign
    poulpy.rotate_assign %mod, %dst, %akm, %scratch {k = 1 : i64} : (!module, memref<!ct>, !akm, !scratch) -> ()
    return
  }

  // CHECK: func @test_rescale
  func.func @test_rescale(%mod: !module, %dst: memref<!ct>, %src: memref<!ct>, %scratch: !scratch) {
    // CHECK: poulpy.rescale
    poulpy.rescale %mod, %dst, %src, %scratch {bits = 40 : i64} : (!module, memref<!ct>, memref<!ct>, !scratch) -> ()
    return
  }

  // CHECK: func @test_rescale_assign
  func.func @test_rescale_assign(%mod: !module, %dst: memref<!ct>, %scratch: !scratch) {
    // CHECK: poulpy.rescale_assign
    poulpy.rescale_assign %mod, %dst, %scratch {bits = 40 : i64} : (!module, memref<!ct>, !scratch) -> ()
    return
  }

  // CHECK: func @test_normalize
  func.func @test_normalize(%mod: !module, %dst: memref<!ct>, %src: memref<!unnorm_ct>, %scratch: !scratch) {
    // CHECK: poulpy.normalize
    poulpy.normalize %mod, %dst, %src, %scratch : (!module, memref<!ct>, memref<!unnorm_ct>, !scratch) -> ()
    return
  }

  // CHECK: func @test_compact_limbs
  func.func @test_compact_limbs(%mod: !module, %dst: memref<!ct>, %src: memref<!ct>, %scratch: !scratch) {
    // CHECK: poulpy.compact_limbs
    poulpy.compact_limbs %mod, %dst, %src, %scratch : (!module, memref<!ct>, memref<!ct>, !scratch) -> ()
    return
  }
}
