// RUN: heir-opt --cheddar-gpu-expand-bootstrap %s | FileCheck %s

module attributes {
  cheddar_gpu.params = #cheddar_gpu.params<
    word = 64,
    log_n = 16,
    ter = [],
    main = [1125899908022273, 1099515691009, 1099523555329, 1099525128193, 1099526176769, 1099529060353, 1099535220737, 1099536138241, 1099537580033, 1099538104321, 1099540725761, 1099540856833, 1099543085057, 36028797019488257, 36028797023420417, 36028797024206849, 36028797025124353, 36028797032202241, 36028797033644033, 36028797037576193, 36028797048324097, 36028797048586241, 36028797049896961, 36028797051863041, 36028797053698049, 36028797054222337],
    aux = [72057594038321153, 72057594040680449, 72057594042646529, 72057594047889409, 72057594057195521, 72057594058375169, 72057594058899457],
    level_config = [[1 : i32, 0 : i32], [2 : i32, 0 : i32], [3 : i32, 0 : i32], [4 : i32, 0 : i32], [5 : i32, 0 : i32], [6 : i32, 0 : i32], [7 : i32, 0 : i32], [8 : i32, 0 : i32], [9 : i32, 0 : i32], [10 : i32, 0 : i32], [11 : i32, 0 : i32], [12 : i32, 0 : i32], [13 : i32, 0 : i32], [14 : i32, 0 : i32], [15 : i32, 0 : i32], [16 : i32, 0 : i32], [17 : i32, 0 : i32], [18 : i32, 0 : i32], [19 : i32, 0 : i32], [20 : i32, 0 : i32], [21 : i32, 0 : i32], [22 : i32, 0 : i32], [23 : i32, 0 : i32], [24 : i32, 0 : i32], [25 : i32, 0 : i32], [26 : i32, 0 : i32]],
    alpha = 7,
    dnum = 0
  >
} {
  // CHECK: func.func @test_expand_eval_mod
  // CHECK-NOT: cheddar_gpu.eval_mod
  // CHECK: call @__cheddar_eval_mod_l21
  // CHECK: return
  func.func @test_expand_eval_mod(
      %out_bx: memref<14x65536xi64>, %out_ax: memref<14x65536xi64>,
      %in_bx: memref<22x65536xi64>, %in_ax: memref<22x65536xi64>,
      %evk_slab: memref<?xi8>) {
    cheddar_gpu.eval_mod %out_bx, %out_ax, %in_bx, %in_ax, %evk_slab
        <{relin_evk_byte_offset = 0 : i64, relin_evk_num_q = 26 : i64, relin_evk_beta = 4 : i64,
          in_form = #cheddar_gpu.form<eval>, out_form = #cheddar_gpu.form<eval>}>
        : (memref<14x65536xi64>, memref<14x65536xi64>, memref<22x65536xi64>, memref<22x65536xi64>, memref<?xi8>) -> ()
    return
  }

  // CHECK: func.func @test_boot_32768_l0_to_l10
  // CHECK-NOT: cheddar_gpu.boot
  // CHECK-NOT: cheddar_gpu.eval_mod
  // CHECK: call @__cheddar_boot_s32768
  // CHECK: return
  func.func @test_boot_32768_l0_to_l10(
      %out_bx: memref<11x65536xi64>, %out_ax: memref<11x65536xi64>,
      %in_bx: memref<1x65536xi64>, %in_ax: memref<1x65536xi64>,
      %consts_slab: memref<?xi8>, %evk_slab: memref<?xi8>) {
    cheddar_gpu.boot
        ins(%in_bx, %in_ax, %consts_slab, %evk_slab
            : memref<1x65536xi64>, memref<1x65536xi64>, memref<?xi8>, memref<?xi8>)
        outs(%out_bx, %out_ax : memref<11x65536xi64>, memref<11x65536xi64>)
        <{in_form = #cheddar_gpu.form<eval>, num_slots = 32768 : i64, out_form = #cheddar_gpu.form<eval>}>
    return
  }

  // CHECK: func.func @test_boot_1024_l0_to_l10
  // CHECK-NOT: cheddar_gpu.boot
  // CHECK-NOT: cheddar_gpu.eval_mod
  // CHECK: call @__cheddar_boot_s1024
  // CHECK: return
  func.func @test_boot_1024_l0_to_l10(
      %out_bx: memref<11x65536xi64>, %out_ax: memref<11x65536xi64>,
      %in_bx: memref<1x65536xi64>, %in_ax: memref<1x65536xi64>,
      %consts_slab: memref<?xi8>, %evk_slab: memref<?xi8>) {
    cheddar_gpu.boot
        ins(%in_bx, %in_ax, %consts_slab, %evk_slab
            : memref<1x65536xi64>, memref<1x65536xi64>, memref<?xi8>, memref<?xi8>)
        outs(%out_bx, %out_ax : memref<11x65536xi64>, memref<11x65536xi64>)
        <{in_form = #cheddar_gpu.form<eval>, num_slots = 1024 : i64, out_form = #cheddar_gpu.form<eval>}>
    return
  }

  // CHECK: func.func @test_boot_32768_l2_to_l8
  // CHECK-NOT: cheddar_gpu.boot
  // CHECK-NOT: cheddar_gpu.eval_mod
  // CHECK: cheddar_gpu.cp_accum_2
  // CHECK: cheddar_gpu.rescale
  // CHECK: cheddar_gpu.cp_accum_2
  // CHECK: cheddar_gpu.rescale
  // CHECK: call @__cheddar_boot_s32768
  // CHECK: cheddar_gpu.cp_accum_2
  // CHECK: cheddar_gpu.rescale
  // CHECK: cheddar_gpu.cp_accum_2
  // CHECK: cheddar_gpu.rescale
  // CHECK: return
  func.func @test_boot_32768_l2_to_l8(
      %out_bx: memref<9x65536xi64>, %out_ax: memref<9x65536xi64>,
      %in_bx: memref<3x65536xi64>, %in_ax: memref<3x65536xi64>,
      %consts_slab: memref<?xi8>, %evk_slab: memref<?xi8>) {
    cheddar_gpu.boot
        ins(%in_bx, %in_ax, %consts_slab, %evk_slab
            : memref<3x65536xi64>, memref<3x65536xi64>, memref<?xi8>, memref<?xi8>)
        outs(%out_bx, %out_ax : memref<9x65536xi64>, memref<9x65536xi64>)
        <{in_form = #cheddar_gpu.form<eval>, num_slots = 32768 : i64, out_form = #cheddar_gpu.form<eval>}>
    return
  }

  // CHECK: func.func private @__cheddar_boot_s32768
  // CHECK: cheddar_gpu.cp_accum_2
  // CHECK: cheddar_gpu.key_switch
  // CHECK: cheddar_gpu.intt
  // CHECK: cheddar_gpu.mod_up_to_max
  // CHECK: cheddar_gpu.ntt
  // CHECK: cheddar_gpu.mul
  // CHECK: cheddar_gpu.cp_accum_add
  // CHECK: cheddar_gpu.mod_down
  // CHECK: cheddar_gpu.hoist_phase
  // CHECK: cheddar_gpu.hconj
  // CHECK: cheddar_gpu.add_2
  // CHECK: cheddar_gpu.sub_2
  // CHECK: cheddar_gpu.mul_imag_unit
  // CHECK: call @__cheddar_eval_mod_l21
  // CHECK: call @__cheddar_eval_mod_l21
  // CHECK: cheddar_gpu.mul_imag_unit
  // CHECK: cheddar_gpu.add_2
  // CHECK: cheddar_gpu.hoist_phase
  // CHECK: return

  // CHECK: func.func private @__cheddar_boot_s1024
  // CHECK: cheddar_gpu.cp_accum_2
  // CHECK: cheddar_gpu.key_switch
  // CHECK: cheddar_gpu.hrot_add
  // CHECK: cheddar_gpu.hoist_phase
  // CHECK: cheddar_gpu.key_switch
  // CHECK: cheddar_gpu.permute_accum_2
  // CHECK: call @__cheddar_eval_mod_l21
  // CHECK: cheddar_gpu.hoist_phase
  // CHECK: cheddar_gpu.hrot_add
  // CHECK: return

  // CHECK: func.func private @__cheddar_eval_mod_l21
  // CHECK: memref.view
  // CHECK: cheddar_gpu.add_const
  // CHECK: cheddar_gpu.tensor
  // CHECK: cheddar_gpu.relinearize_rescale
  // CHECK: cheddar_gpu.cp_accum_2
  // CHECK: cheddar_gpu.tensor
  // CHECK: cheddar_gpu.add_const
  // CHECK: cheddar_gpu.relinearize_rescale
  // CHECK: return
}
