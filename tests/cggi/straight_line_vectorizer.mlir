// TODO(#519): disable FileChecks until nondeterminism issues are resolved
// RUN: heir-opt --straight-line-vectorize %s

#encoding = #lwe.unspecified_bit_field_encoding<cleartext_bitwidth = 3>
!ct_ty = !lwe.lwe_ciphertext<encoding = #encoding>
!pt_ty = !lwe.lwe_plaintext<encoding = #encoding>

// CHECK-LABEL: add_one
// CHECK-COUNT-9: cggi.lut3
// CHECK: cggi.lut3 %[[arg1:.*]], %[[arg2:.*]], %[[arg3:.*]] {lookup_table = 105 : ui8} : tensor<6x!lwe.lwe_ciphertext
func.func @add_one(%arg0: tensor<8x!ct_ty>) -> tensor<8x!ct_ty> {
  %true = arith.constant true
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %extracted = tensor.extract %arg0[%c0] : tensor<8x!ct_ty>
  %extracted_0 = tensor.extract %arg0[%c1] : tensor<8x!ct_ty>
  %extracted_1 = tensor.extract %arg0[%c2] : tensor<8x!ct_ty>
  %extracted_2 = tensor.extract %arg0[%c3] : tensor<8x!ct_ty>
  %extracted_3 = tensor.extract %arg0[%c4] : tensor<8x!ct_ty>
  %extracted_4 = tensor.extract %arg0[%c5] : tensor<8x!ct_ty>
  %extracted_5 = tensor.extract %arg0[%c6] : tensor<8x!ct_ty>
  %extracted_6 = tensor.extract %arg0[%c7] : tensor<8x!ct_ty>
  %0 = lwe.encode %true {encoding = #encoding} : i1 to !pt_ty
  %1 = lwe.trivial_encrypt %0 : !pt_ty to !ct_ty
  %2 = lwe.encode %false {encoding = #encoding} : i1 to !pt_ty
  %3 = lwe.trivial_encrypt %2 : !pt_ty to !ct_ty
  %4 = cggi.lut3 %extracted, %1, %3 {lookup_table = 8 : ui8} : !ct_ty
  %5 = cggi.lut3 %4, %extracted_0, %3 {lookup_table = 150 : ui8} : !ct_ty
  %6 = cggi.lut3 %4, %extracted_0, %3 {lookup_table = 23 : ui8} : !ct_ty
  %7 = cggi.lut3 %6, %extracted_1, %3 {lookup_table = 43 : ui8} : !ct_ty
  %8 = cggi.lut3 %7, %extracted_2, %3 {lookup_table = 43 : ui8} : !ct_ty
  %9 = cggi.lut3 %8, %extracted_3, %3 {lookup_table = 43 : ui8} : !ct_ty
  %10 = cggi.lut3 %9, %extracted_4, %3 {lookup_table = 43 : ui8} : !ct_ty
  %11 = cggi.lut3 %10, %extracted_5, %3 {lookup_table = 105 : ui8} : !ct_ty
  %12 = cggi.lut3 %10, %extracted_5, %3 {lookup_table = 43 : ui8} : !ct_ty
  %13 = cggi.lut3 %12, %extracted_6, %3 {lookup_table = 105 : ui8} : !ct_ty
  %14 = cggi.lut3 %extracted, %1, %3 {lookup_table = 6 : ui8} : !ct_ty
  %15 = cggi.lut3 %6, %extracted_1, %3 {lookup_table = 105 : ui8} : !ct_ty
  %16 = cggi.lut3 %7, %extracted_2, %3 {lookup_table = 105 : ui8} : !ct_ty
  %17 = cggi.lut3 %8, %extracted_3, %3 {lookup_table = 105 : ui8} : !ct_ty
  %18 = cggi.lut3 %9, %extracted_4, %3 {lookup_table = 105 : ui8} : !ct_ty
  %from_elements = tensor.from_elements %13, %11, %18, %17, %16, %15, %5, %14 : tensor<8x!ct_ty>
  return %from_elements : tensor<8x!ct_ty>
}

// CHECK-LABEL: require_post_pass_toposort
func.func @require_post_pass_toposort(%arg0: tensor<8x!ct_ty>) -> tensor<8x!ct_ty> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %0 = tensor.extract %arg0[%c0] : tensor<8x!ct_ty>
  %1 = tensor.extract %arg0[%c1] : tensor<8x!ct_ty>
  %2 = tensor.extract %arg0[%c2] : tensor<8x!ct_ty>
  %3 = tensor.extract %arg0[%c3] : tensor<8x!ct_ty>
  %4 = tensor.extract %arg0[%c4] : tensor<8x!ct_ty>
  %5 = tensor.extract %arg0[%c5] : tensor<8x!ct_ty>
  %6 = tensor.extract %arg0[%c6] : tensor<8x!ct_ty>
  %7 = tensor.extract %arg0[%c7] : tensor<8x!ct_ty>

  // Four ops that can be vectorized
  %r1 = cggi.lut3 %0, %1, %2 {lookup_table = 8 : ui8} : !ct_ty
  %r2 = cggi.lut3 %3, %4, %5 {lookup_table = 8 : ui8} : !ct_ty
  %r3 = cggi.lut3 %4, %5, %6 {lookup_table = 8 : ui8} : !ct_ty
  %r4 = cggi.lut3 %5, %6, %7 {lookup_table = 8 : ui8} : !ct_ty

  // A non-vectorizable op that uses one of the results
  %x = cggi.not %r4 : !ct_ty

  // Four more ops that can be vectorized
  %r5 = cggi.lut3 %0, %3, %1 {lookup_table = 8 : ui8} : !ct_ty
  %r6 = cggi.lut3 %2, %5, %6 {lookup_table = 8 : ui8} : !ct_ty
  %r7 = cggi.lut3 %7, %1, %6 {lookup_table = 8 : ui8} : !ct_ty
  %r8 = cggi.lut3 %3, %6, %0 {lookup_table = 8 : ui8} : !ct_ty

  // The not op has to occur after the lut3s, since it depends on one of the
  // results.

  // CHECK: cggi.lut3 %[[arg1:.*]], %[[arg2:.*]], %[[arg3:.*]] {lookup_table = 8 : ui8} : tensor<8x!lwe.lwe_ciphertext
  // CHECK: cggi.not

  %from_elements = tensor.from_elements %r1, %r2, %r3, %r4, %r5, %r6, %r7, %x : tensor<8x!ct_ty>
  return %from_elements : tensor<8x!ct_ty>
}

// CHECK-LABEL: transitive_dep_splits_level
func.func @transitive_dep_splits_level(%arg0: tensor<8x!ct_ty>) -> tensor<8x!ct_ty> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %0 = tensor.extract %arg0[%c0] : tensor<8x!ct_ty>
  %1 = tensor.extract %arg0[%c1] : tensor<8x!ct_ty>
  %2 = tensor.extract %arg0[%c2] : tensor<8x!ct_ty>
  %3 = tensor.extract %arg0[%c3] : tensor<8x!ct_ty>
  %4 = tensor.extract %arg0[%c4] : tensor<8x!ct_ty>
  %5 = tensor.extract %arg0[%c5] : tensor<8x!ct_ty>
  %6 = tensor.extract %arg0[%c6] : tensor<8x!ct_ty>
  %7 = tensor.extract %arg0[%c7] : tensor<8x!ct_ty>

  // Four ops that can be vectorized
  %r1 = cggi.lut3 %0, %1, %2 {lookup_table = 8 : ui8} : !ct_ty
  %r2 = cggi.lut3 %3, %4, %5 {lookup_table = 8 : ui8} : !ct_ty
  %r3 = cggi.lut3 %4, %5, %6 {lookup_table = 8 : ui8} : !ct_ty
  %r4 = cggi.lut3 %5, %6, %7 {lookup_table = 8 : ui8} : !ct_ty

  // A non-vectorizable op that uses one of the results
  %n1 = cggi.not %r1 : !ct_ty
  %n2 = cggi.not %r2 : !ct_ty
  %n3 = cggi.not %r3 : !ct_ty
  %n4 = cggi.not %r4 : !ct_ty

  // Four more ops that can be vectorized
  %r5 = cggi.lut3 %0, %n1, %1 {lookup_table = 8 : ui8} : !ct_ty
  %r6 = cggi.lut3 %2, %n2, %6 {lookup_table = 8 : ui8} : !ct_ty
  %r7 = cggi.lut3 %7, %n3, %6 {lookup_table = 8 : ui8} : !ct_ty
  %r8 = cggi.lut3 %3, %n4, %0 {lookup_table = 8 : ui8} : !ct_ty

  // The slice analysis ensures these are split into two levels of 4 ops each.
  // CHECK: cggi.lut3 %[[arg1:.*]], %[[arg2:.*]], %[[arg3:.*]] {lookup_table = 8 : ui8} : tensor<4x!lwe.lwe_ciphertext
  // CHECK-COUNT-4: cggi.not
  // CHECK: cggi.lut3 %[[arg1:.*]], %[[arg2:.*]], %[[arg3:.*]] {lookup_table = 8 : ui8} : tensor<4x!lwe.lwe_ciphertext

  %from_elements = tensor.from_elements %r1, %r2, %r3, %r4, %r5, %r6, %r7, %r8 : tensor<8x!ct_ty>
  return %from_elements : tensor<8x!ct_ty>
}
