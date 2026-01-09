// RUN: heir-opt -fold-constant-tensors -forward-insert-to-extract %s | FileCheck %s


!cc = !openfhe.crypto_context
!ct = !openfhe.ciphertext
!pt = !openfhe.plaintext


//  CHECK: @successful_forwarding
//  CHECK-SAME:  (%[[ARG0:.*]]: !cc,


func.func @successful_forwarding(%arg0: !cc, %arg1: tensor<1x16x!ct>, %arg2: tensor<1x16x!ct>, %arg3: tensor<16xf64>, %arg4: tensor<16xf64>) -> tensor<1x16x!ct> {

  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
  %c1 = arith.constant 1 : index

  //  CHECK-NEXT: %[[EXTRACTED:.*]] = tensor.extract
  %extracted = tensor.extract %arg1[%c0, %c0] : tensor<1x16x!ct>
  //  CHECK-NEXT: %[[EXTRACTED0:.*]] = tensor.extract
  %extracted_0 = tensor.extract %arg2[%c0, %c0] : tensor<1x16x!ct>
  //  CHECK-NEXT: %[[VAL0:.*]] = openfhe.make_ckks_packed_plaintext %[[ARG0]]
  %0 = openfhe.make_ckks_packed_plaintext %arg0, %arg3 : (!cc, tensor<16xf64>) -> !pt
  //  CHECK-NEXT: %[[VAL1:.*]] = openfhe.mul_plain %[[ARG0]], %[[EXTRACTED]], %[[VAL0]]
  %1 = openfhe.mul_plain %arg0, %extracted, %0 : (!cc, !ct, !pt) -> !ct
  //  CHECK-NEXT: %[[VAL2:.*]] = openfhe.add %[[ARG0]], %[[EXTRACTED0]], %[[VAL1]]
  %2 = openfhe.add %arg0, %extracted_0, %1 : (!cc, !ct, !ct) -> !ct

  //  CHECK-NEXT: %[[INSERTED0:.*]] = tensor.insert %[[VAL2]]
  %inserted = tensor.insert %2 into %arg2[%c0, %c0] : tensor<1x16x!ct>

  //  CHECK-NEXT: %[[EXTRACTED1:.*]] = tensor.extract
  %extracted_1 = tensor.extract %arg1[%c0, %c1] : tensor<1x16x!ct>
  //  CHECK-NOT: tensor.extract %[[INSERTED0]]
  %extracted_2 = tensor.extract %inserted[%c0, %c0] : tensor<1x16x!ct>
  //  CHECK-NEXT: %[[VAL3:.*]] = openfhe.make_ckks_packed_plaintext
  %3 = openfhe.make_ckks_packed_plaintext %arg0, %arg4 : (!cc, tensor<16xf64>) -> !pt
  //  CHECK-NEXT: %[[VAL4:.*]] = openfhe.mul_plain
  %4 = openfhe.mul_plain %arg0, %extracted_1, %3 : (!cc, !ct, !pt) -> !ct
  //  CHECK-NEXT: %[[VAL5:.*]] = openfhe.add
  %5 = openfhe.add %arg0, %extracted_2, %4 : (!cc, !ct, !ct) -> !ct
  //  CHECK-NEXT: %[[INSERTED1:.*]] = tensor.insert
  %inserted_3 = tensor.insert %5 into %inserted[%c0, %c0] : tensor<1x16x!ct>
  //  CHECK-NEXT: return %[[INSERTED1]]
  return %inserted_3 : tensor<1x16x!ct>
}


//hits def == nullptr
//  CHECK: @forward_from_func_arg
//  CHECK-SAME:  (%[[ARG0:.*]]: !cc,

func.func @forward_from_func_arg(%arg0: !cc, %arg1: tensor<1x16x!ct>, %arg2: tensor<1x16x!ct>)-> !ct {
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  //  CHECK-NEXT: %[[EXTRACTED:.*]] = tensor.extract
  %extracted = tensor.extract %arg1[%c0, %c0] : tensor<1x16x!ct>

  return %extracted : !ct
}

//  CHECK: @forwarding_with_an_insert_in_between
//  CHECK-SAME:  (%[[ARG0:.*]]: !cc,

func.func @forwarding_with_an_insert_in_between(%arg0: !cc, %arg1: tensor<1x16x!ct>, %arg2: tensor<1x16x!ct>, %arg3: tensor<16xf64> )-> !ct {

  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index

  //  CHECK-NEXT: %[[EXTRACTED:.*]] = tensor.extract
  %extracted = tensor.extract %arg1[%c0, %c0] : tensor<1x16x!ct>
  //  CHECK-NEXT: %[[EXTRACTED0:.*]] = tensor.extract
  %extracted_0 = tensor.extract %arg2[%c0, %c0] : tensor<1x16x!ct>
  //  CHECK-NEXT: %[[VAL0:.*]] = openfhe.make_ckks_packed_plaintext %[[ARG0]]
  %0 = openfhe.make_ckks_packed_plaintext %arg0, %arg3 : (!cc, tensor<16xf64>) -> !pt
  //  CHECK-NEXT: %[[VAL1:.*]] = openfhe.mul_plain %[[ARG0]], %[[EXTRACTED]], %[[VAL0]]
  %1 = openfhe.mul_plain %arg0, %extracted, %0 : (!cc, !ct, !pt) -> !ct
  //  CHECK-NEXT: %[[VAL2:.*]] = openfhe.add %[[ARG0]], %[[EXTRACTED0]], %[[VAL1]]
  %2 = openfhe.add %arg0, %extracted_0, %1 : (!cc, !ct, !ct) -> !ct
  //  CHECK-NEXT: %[[VALA2:.*]] = openfhe.add %[[ARG0]], %[[EXTRACTED0]], %[[VAL2]]
  %a2 = openfhe.add %arg0, %extracted_0, %2 : (!cc, !ct, !ct) -> !ct
  //  CHECK-NOT: tensor.insert %[[VAL2]]
  %inserted = tensor.insert %2 into %arg2[%c0, %c0] : tensor<1x16x!ct>
  //  CHECK-NOT: tensor.insert %[[VALA2]]
  %inserted_1 = tensor.insert %a2 into %arg1[%c0, %c0] : tensor<1x16x!ct>

  //  CHECK-NOT: tensor.extract
  %extracted_2 = tensor.extract %inserted_1[%c0, %c0] : tensor<1x16x!ct>
  // CHECK: return %[[VALA2]]
  return %extracted_2 : !ct
}

//  CHECK: @forwarding_with_an_operation_in_between
//  CHECK-SAME:  (%[[ARG0:.*]]: !cc,

func.func @forwarding_with_an_operation_in_between(%arg0: !cc, %arg1: tensor<1x16x!ct>, %arg2: tensor<1x16x!ct>, %arg3: tensor<16xf64>, %arg4: i1 )-> !ct {

  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index

  //  CHECK-NEXT: %[[EXTRACTED:.*]] = tensor.extract
  %extracted = tensor.extract %arg1[%c0, %c0] : tensor<1x16x!ct>
  //  CHECK-NEXT: %[[EXTRACTED0:.*]] = tensor.extract
  %extracted_0 = tensor.extract %arg2[%c0, %c0] : tensor<1x16x!ct>
  //  CHECK-NEXT: %[[VAL0:.*]] = openfhe.make_ckks_packed_plaintext %[[ARG0]]
  %0 = openfhe.make_ckks_packed_plaintext %arg0, %arg3 : (!cc, tensor<16xf64>) -> !pt
  //  CHECK-NEXT: %[[VAL1:.*]] = openfhe.mul_plain %[[ARG0]], %[[EXTRACTED]], %[[VAL0]]
  %1 = openfhe.mul_plain %arg0, %extracted, %0 : (!cc, !ct, !pt) -> !ct
  //  CHECK-NEXT: %[[VAL2:.*]] = openfhe.add %[[ARG0]], %[[EXTRACTED0]], %[[VAL1]]
  %2 = openfhe.add %arg0, %extracted_0, %1 : (!cc, !ct, !ct) -> !ct

  //  CHECK-NOT: %[[INSERTED0:.*]] = tensor.insert %[[VAL2]]
  %inserted = tensor.insert %2 into %arg2[%c0, %c0] : tensor<1x16x!ct>

  scf.if %arg4 {
    //  CHECK-NOT: %[[VALa2:.*]] = openfhe.add %[[ARG0]], %[[EXTRACTED0]], %[[VAL2]]
    %a2 = openfhe.add %arg0, %extracted_0, %2 : (!cc, !ct, !ct) -> !ct
    //  CHECK-NOT: tensor.insert %[[VAL1]]
    %inserted_1 = tensor.insert %a2 into %arg2[%c0, %c0] : tensor<1x16x!ct>
  }
    //  CHECK-NOT: tensor.extract
  %extracted_2 = tensor.extract %inserted[%c0, %c0] : tensor<1x16x!ct>
  return %extracted_2 : !ct
}


//  CHECK: @two_extracts_both_forwarded
//  CHECK-SAME:  (%[[ARG0:.*]]: !cc,

func.func @two_extracts_both_forwarded(%arg0: !cc, %arg1: tensor<1x16x!ct>, %arg2: tensor<1x16x!ct>, %arg3: tensor<16xf64>) -> !ct {

  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index

  //  CHECK-NEXT: %[[EXTRACTED:.*]] = tensor.extract
  %extracted = tensor.extract %arg1[%c0, %c0] : tensor<1x16x!ct>
  //  CHECK-NEXT: %[[EXTRACTED0:.*]] = tensor.extract
  %extracted_0 = tensor.extract %arg2[%c0, %c0] : tensor<1x16x!ct>
  //  CHECK-NEXT: %[[VAL0:.*]] = openfhe.make_ckks_packed_plaintext %[[ARG0]]
  %0 = openfhe.make_ckks_packed_plaintext %arg0, %arg3 : (!cc, tensor<16xf64>) -> !pt
  //  CHECK-NEXT: %[[VAL1:.*]] = openfhe.mul_plain %[[ARG0]], %[[EXTRACTED]], %[[VAL0]]
  %1 = openfhe.mul_plain %arg0, %extracted, %0 : (!cc, !ct, !pt) -> !ct
  //  CHECK-NEXT: %[[VAL2:.*]] = openfhe.add %[[ARG0]], %[[EXTRACTED0]], %[[VAL1]]
  %2 = openfhe.add %arg0, %extracted_0, %1 : (!cc, !ct, !ct) -> !ct

  %inserted = tensor.insert %2 into %arg2[%c0, %c0] : tensor<1x16x!ct>

  //  CHECK-NOT: tensor.extract
  %extracted_1 = tensor.extract %inserted[%c0, %c0] : tensor<1x16x!ct>
  //  CHECK-NOT: tensor.extract
  %extracted_2 = tensor.extract %inserted[%c0, %c0] : tensor<1x16x!ct>
  // CHECK: openfhe.add %[[ARG0]], %[[VAL2]], %[[VAL2]]
  %3 = openfhe.add %arg0, %extracted_1, %extracted_2 : (!cc, !ct, !ct) -> !ct
  return %3: !ct
}

// Tests forwarding an extract after a chain of insertions.

// CHECK: @insert_chain_forwarded
// CHECK-SAME:  (%[[ARG1:.*]]: tensor<2x[[ct_ty:.*]]>,
// CHECK-SAME: %[[ct:.*]]: [[ct_ty]], %[[ct_1:.*]]: [[ct_ty]]) -> [[ct_ty]]
func.func @insert_chain_forwarded(%arg1: tensor<2x!ct>, %ct: !ct, %ct_1: !ct) -> !ct {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %inserted = tensor.insert %ct into %arg1[%c0] : tensor<2x!ct>
  %inserted_1 = tensor.insert %ct_1 into %inserted[%c1] : tensor<2x!ct>
  %extracted = tensor.extract %inserted_1[%c0] : tensor<2x!ct>
  // CHECK: return %[[ct]]
  return %extracted : !ct
}

// Tests forwarding a value from a constant through use-def chain.

// CHECK: @extract_from_constant
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32)
func.func @extract_from_constant(%arg0: i32, %arg1: i32) -> (i32, i32) {
  // CHECK: %[[c3_i32:.*]] = arith.constant 3 : i32
  %const = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %inserted = tensor.insert %arg0 into %const[%c0] : tensor<3xi32>
  %inserted_1 = tensor.insert %arg1 into %inserted[%c1] : tensor<3xi32>
  %extracted_0 = tensor.extract %inserted_1[%c0] : tensor<3xi32>
  %extracted_2 = tensor.extract %inserted_1[%c2] : tensor<3xi32>
  // CHECK: return %[[ARG0]], %[[c3_i32]]
  return %extracted_0, %extracted_2 : i32, i32
}

// Tests forwarding a value from a from_elements op through use-def chain.

// CHECK: @extract_from_elements
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32)
func.func @extract_from_elements(%arg0: i32, %arg1: i32, %arg2: i32) -> (i32, i32) {
  %const = tensor.from_elements %arg0, %arg0, %arg0 : tensor<3xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %inserted = tensor.insert %arg1 into %const[%c1] : tensor<3xi32>
  %inserted_1 = tensor.insert %arg2 into %inserted[%c2] : tensor<3xi32>
  %extracted_0 = tensor.extract %inserted_1[%c0] : tensor<3xi32>
  %extracted_2 = tensor.extract %inserted_1[%c2] : tensor<3xi32>
  // CHECK: return %[[ARG0]], %[[ARG2]]
  return %extracted_0, %extracted_2 : i32, i32
}
