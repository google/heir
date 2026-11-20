// RUN: heir-opt --verify-roundtrip %s | FileCheck %s

// CHECK: func.func @test_mod_up_to_max
// CHECK: cheddar_gpu.mod_up_to_max %{{.*}}, %{{.*}}{{.*}} : (memref<4x65536xi32>, memref<1x65536xi32>) -> ()
func.func @test_mod_up_to_max(%out: memref<4x65536xi32>, %in: memref<1x65536xi32>) {
  cheddar_gpu.mod_up_to_max %out, %in : (memref<4x65536xi32>, memref<1x65536xi32>) -> ()
  return
}

// CHECK: func.func @test_zero_pad
// CHECK: cheddar_gpu.zero_pad %{{.*}} <{{.*}}limb_range = [1, 3]{{.*}}> : (memref<4x65536xi32>) -> ()
func.func @test_zero_pad(%out: memref<4x65536xi32>) {
  cheddar_gpu.zero_pad %out <{limb_range = array<i64: 1, 3>}> : (memref<4x65536xi32>) -> ()
  return
}

// CHECK: func.func @test_hoist_bs_fused
// CHECK: cheddar_gpu.hoist_bs_fused out_bx(%{{.*}}), out_ax(%{{.*}}), mod_up(%{{.*}}), %{{.*}}, %{{.*}} <{{.*}}galois_factors = [5]{{.*}}> : (memref<3x65536xi32>, memref<3x65536xi32>, memref<3x65536xi32>, memref<1024xi8>, memref<2x65536xi32>) -> ()
func.func @test_hoist_bs_fused(
    %out_bx: memref<3x65536xi32>, %out_ax: memref<3x65536xi32>,
    %mod_up: memref<3x65536xi32>, %evk_slab: memref<1024xi8>,
    %in_bx_pseudo_modup: memref<2x65536xi32>) {
  cheddar_gpu.hoist_bs_fused out_bx(%out_bx), out_ax(%out_ax), mod_up(%mod_up), %evk_slab, %in_bx_pseudo_modup <{
    galois_factors = array<i64: 5>,
    evk_byte_offsets = array<i64: 0>,
    evk_num_q = array<i64: 2>,
    num_q = 2 : i64,
    num_aux = 1 : i64,
    prime_offset = 0 : i64
  }> : (memref<3x65536xi32>, memref<3x65536xi32>, memref<3x65536xi32>, memref<1024xi8>, memref<2x65536xi32>) -> ()
  return
}

// CHECK: func.func @test_hoist_gs_fused
// CHECK: cheddar_gpu.hoist_gs_fused out_bx(%{{.*}}), out_ax(%{{.*}}), in_bx(%{{.*}}), in_ax(%{{.*}}), %{{.*}} <{{.*}}pt_word_offsets = [0]{{.*}}> : (memref<3x65536xi32>, memref<3x65536xi32>, memref<3x65536xi32>, memref<3x65536xi32>, memref<1024xi8>) -> ()
func.func @test_hoist_gs_fused(
    %out_bx: memref<3x65536xi32>, %out_ax: memref<3x65536xi32>,
    %in_bx: memref<3x65536xi32>, %in_ax: memref<3x65536xi32>,
    %consts_slab: memref<1024xi8>) {
  cheddar_gpu.hoist_gs_fused out_bx(%out_bx), out_ax(%out_ax), in_bx(%in_bx), in_ax(%in_ax), %consts_slab <{
    pt_word_offsets = array<i64: 0>,
    num_q = 2 : i64,
    num_aux = 1 : i64,
    prime_offset = 0 : i64,
    pt_limbs = 3 : i64
  }> : (memref<3x65536xi32>, memref<3x65536xi32>, memref<3x65536xi32>, memref<3x65536xi32>, memref<1024xi8>) -> ()
  return
}

// CHECK: func.func @test_eval_mod
// CHECK: cheddar_gpu.eval_mod ins(%{{.*}}, %{{.*}}, %{{.*}} : memref<22x65536xi32>, memref<22x65536xi32>, memref<1024xi8>) outs(%{{.*}}, %{{.*}} : memref<14x65536xi32>, memref<14x65536xi32>) <{{.*}}relin_evk_num_q = 22{{.*}}>
func.func @test_eval_mod(
    %out_bx: memref<14x65536xi32>, %out_ax: memref<14x65536xi32>,
    %in_bx: memref<22x65536xi32>, %in_ax: memref<22x65536xi32>,
    %evk_slab: memref<1024xi8>) {
  cheddar_gpu.eval_mod ins(%in_bx, %in_ax, %evk_slab : memref<22x65536xi32>, memref<22x65536xi32>, memref<1024xi8>) outs(%out_bx, %out_ax : memref<14x65536xi32>, memref<14x65536xi32>) <{
    relin_evk_byte_offset = 0 : i64,
    relin_evk_num_q = 22 : i64,
    relin_evk_beta = 1 : i64
  }>
  return
}

// CHECK: func.func @test_hoist_phase
// CHECK: cheddar_gpu.hoist_phase ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : memref<26x65536xi32>, memref<26x65536xi32>, memref<1024xi8>, memref<1024xi8>) outs(%{{.*}}, %{{.*}} : memref<25x65536xi32>, memref<25x65536xi32>) <{{.*}}kind = cts{{.*}}phase = 0{{.*}}>
func.func @test_hoist_phase(
    %out_bx: memref<25x65536xi32>, %out_ax: memref<25x65536xi32>,
    %in_bx: memref<26x65536xi32>, %in_ax: memref<26x65536xi32>,
    %consts_slab: memref<1024xi8>, %evk_slab: memref<1024xi8>) {
  cheddar_gpu.hoist_phase ins(%in_bx, %in_ax, %consts_slab, %evk_slab : memref<26x65536xi32>, memref<26x65536xi32>, memref<1024xi8>, memref<1024xi8>) outs(%out_bx, %out_ax : memref<25x65536xi32>, memref<25x65536xi32>) <{
    kind = #cheddar_gpu.hoist_phase_kind<cts>,
    phase = 0 : i64,
    num_slots = 256 : i64
  }>
  return
}

// CHECK: func.func @test_boot
// CHECK: cheddar_gpu.boot ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : memref<14x65536xi64>, memref<14x65536xi64>, memref<1024xi8>, memref<1024xi8>) outs(%{{.*}}, %{{.*}} : memref<11x65536xi64>, memref<11x65536xi64>) <{{.*}}num_slots = 256{{.*}}>
func.func @test_boot(
    %out_bx: memref<11x65536xi64>, %out_ax: memref<11x65536xi64>,
    %in_bx: memref<14x65536xi64>, %in_ax: memref<14x65536xi64>,
    %consts_slab: memref<1024xi8>, %evk_slab: memref<1024xi8>) {
  cheddar_gpu.boot ins(%in_bx, %in_ax, %consts_slab, %evk_slab : memref<14x65536xi64>, memref<14x65536xi64>, memref<1024xi8>, memref<1024xi8>) outs(%out_bx, %out_ax : memref<11x65536xi64>, memref<11x65536xi64>) <{
    num_slots = 256 : i64
  }>
  return
}
