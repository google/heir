#unspecified_bit_field_encoding = #lwe.unspecified_bit_field_encoding<cleartext_bitwidth = 1>
module attributes {tf_saved_model.semantics} {
  func.func @main(%arg0: memref<1x1x8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>) -> memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>> {
    %false = arith.constant false
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c9 = arith.constant 9 : index
    %c10 = arith.constant 10 : index
    %c11 = arith.constant 11 : index
    %c12 = arith.constant 12 : index
    %c13 = arith.constant 13 : index
    %c14 = arith.constant 14 : index
    %c15 = arith.constant 15 : index
    %c16 = arith.constant 16 : index
    %c17 = arith.constant 17 : index
    %c18 = arith.constant 18 : index
    %c19 = arith.constant 19 : index
    %c20 = arith.constant 20 : index
    %c21 = arith.constant 21 : index
    %c22 = arith.constant 22 : index
    %c23 = arith.constant 23 : index
    %c24 = arith.constant 24 : index
    %c25 = arith.constant 25 : index
    %c26 = arith.constant 26 : index
    %c27 = arith.constant 27 : index
    %c28 = arith.constant 28 : index
    %c29 = arith.constant 29 : index
    %c30 = arith.constant 30 : index
    %c31 = arith.constant 31 : index
    %0 = memref.load %arg0[%c0, %c0, %c0] : memref<1x1x8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %1 = cggi.not %0 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %2 = memref.load %arg0[%c0, %c0, %c7] : memref<1x1x8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %3 = cggi.not %2 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %4 = memref.load %arg0[%c0, %c0, %c6] : memref<1x1x8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %5 = cggi.and %4, %3 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %6 = memref.load %arg0[%c0, %c0, %c4] : memref<1x1x8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %7 = memref.load %arg0[%c0, %c0, %c5] : memref<1x1x8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %8 = cggi.and %6, %7 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %9 = cggi.and %8, %5 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %10 = memref.load %arg0[%c0, %c0, %c2] : memref<1x1x8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %11 = memref.load %arg0[%c0, %c0, %c3] : memref<1x1x8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %12 = cggi.and %10, %11 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %13 = memref.load %arg0[%c0, %c0, %c1] : memref<1x1x8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %14 = cggi.and %13, %0 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %15 = cggi.and %14, %12 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %16 = cggi.nand %15, %9 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %17 = cggi.and %16, %3 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %18 = cggi.and %17, %2 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %19 = cggi.xor %13, %0 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %20 = cggi.xor %14, %10 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %21 = cggi.and %14, %10 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %22 = cggi.xor %21, %11 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %23 = cggi.xor %15, %6 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %24 = cggi.and %15, %6 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %25 = cggi.xor %24, %7 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %26 = cggi.and %15, %8 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %27 = cggi.xor %26, %4 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %28 = cggi.and %26, %4 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %29 = cggi.xor %28, %3 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %30 = cggi.nor %16, %2 : !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %alloc = memref.alloc() : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %1, %alloc[%c0, %c0, %c0] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %19, %alloc[%c0, %c0, %c1] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %20, %alloc[%c0, %c0, %c2] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %22, %alloc[%c0, %c0, %c3] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %23, %alloc[%c0, %c0, %c4] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %25, %alloc[%c0, %c0, %c5] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %27, %alloc[%c0, %c0, %c6] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %29, %alloc[%c0, %c0, %c7] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %30, %alloc[%c0, %c0, %c8] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %18, %alloc[%c0, %c0, %c9] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %18, %alloc[%c0, %c0, %c10] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %18, %alloc[%c0, %c0, %c11] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %18, %alloc[%c0, %c0, %c12] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %18, %alloc[%c0, %c0, %c13] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %18, %alloc[%c0, %c0, %c14] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %18, %alloc[%c0, %c0, %c15] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %31 = lwe.encode %false {encoding = #unspecified_bit_field_encoding} : i1 to !lwe.lwe_plaintext<encoding = #unspecified_bit_field_encoding>
    %32 = lwe.trivial_encrypt %31 : !lwe.lwe_plaintext<encoding = #unspecified_bit_field_encoding> to !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    memref.store %32, %alloc[%c0, %c0, %c16] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %32, %alloc[%c0, %c0, %c17] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %32, %alloc[%c0, %c0, %c18] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %32, %alloc[%c0, %c0, %c19] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %32, %alloc[%c0, %c0, %c20] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %32, %alloc[%c0, %c0, %c21] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %32, %alloc[%c0, %c0, %c22] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %32, %alloc[%c0, %c0, %c23] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %32, %alloc[%c0, %c0, %c24] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %32, %alloc[%c0, %c0, %c25] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %32, %alloc[%c0, %c0, %c26] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %32, %alloc[%c0, %c0, %c27] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %32, %alloc[%c0, %c0, %c28] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %32, %alloc[%c0, %c0, %c29] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %32, %alloc[%c0, %c0, %c30] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %32, %alloc[%c0, %c0, %c31] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    return %alloc : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
  }
}
