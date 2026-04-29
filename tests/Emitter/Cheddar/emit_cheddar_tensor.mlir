// RUN: heir-translate --emit-cheddar %s | FileCheck %s

// Test tensor.empty
// CHECK: void test_tensor_empty
func.func @test_tensor_empty(%ctx: !cheddar.context) {
  // CHECK: std::vector<Ct> [[VEC:.*]];
  // CHECK-NEXT: [[VEC]].resize(4);
  %empty = tensor.empty() : tensor<4x!cheddar.ciphertext>
  return
}

// Test dense splat constants lower to sized vector constructors instead of
// enormous initializer lists.
// CHECK: {{.*}} test_dense_splats
func.func @test_dense_splats() {
  // CHECK: std::vector<double> [[FLOATS:.*]](8, 0{{(\.0+)?}});
  %floats = arith.constant dense<0.0> : tensor<8xf32>
  // CHECK: std::vector<int64_t> [[INTS:.*]](6, 7);
  %ints = arith.constant dense<7> : tensor<6xindex>
  return
}

// Test tensor.from_elements with ciphertexts (move-only)
// CHECK: {{.*}} test_from_elements
func.func @test_from_elements(
    %ctx: !cheddar.context,
    %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext) -> tensor<2x!cheddar.ciphertext> {
  // CHECK: std::vector<Ct> [[VEC:.*]];
  // CHECK-NEXT: [[VEC]].reserve(2);
  // CHECK: Copy(
  // CHECK: [[VEC]].emplace_back(std::move(
  // CHECK: Copy(
  // CHECK: [[VEC]].emplace_back(std::move(
  %t = tensor.from_elements %ct0, %ct1 : tensor<2x!cheddar.ciphertext>
  return %t : tensor<2x!cheddar.ciphertext>
}

// Test tensor.extract with ciphertexts (reference)
// CHECK: {{.*}} test_extract
func.func @test_extract(
    %ctx: !cheddar.context,
    %t: tensor<4x!cheddar.ciphertext>) -> !cheddar.ciphertext {
  %c1 = arith.constant 1 : index
  // CHECK: auto& [[RES:.*]] = {{.*}}[{{.*}}];
  %elem = tensor.extract %t[%c1] : tensor<4x!cheddar.ciphertext>
  return %elem : !cheddar.ciphertext
}

// Test tensor.insert with ciphertexts (in-place move)
// CHECK: {{.*}} test_insert
func.func @test_insert(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext,
    %t: tensor<4x!cheddar.ciphertext>) -> tensor<4x!cheddar.ciphertext> {
  %c2 = arith.constant 2 : index
  // CHECK: Copy(
  // CHECK: {{.*}}[{{.*}}] = std::move(
  %result = tensor.insert %ct into %t[%c2] : tensor<4x!cheddar.ciphertext>
  return %result : tensor<4x!cheddar.ciphertext>
}

// Test tensor.insert into tensor.empty with ciphertexts. This should not deep
// copy the uninitialized empty tensor elements.
// CHECK: {{.*}} test_insert_into_empty
func.func @test_insert_into_empty(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext) -> tensor<1x!cheddar.ciphertext> {
  %empty = tensor.empty() : tensor<1x!cheddar.ciphertext>
  %c0 = arith.constant 0 : index
  // CHECK: std::vector<Ct> [[EMPTY:.*]];
  // CHECK-NEXT: [[EMPTY]].resize(1);
  // CHECK: std::vector<Ct> [[RES:.*]];
  // CHECK-NEXT: [[RES]].resize(1);
  // CHECK-NOT: [[EMPTY]][i]
  // CHECK: Copy(
  // CHECK: [[RES]][{{.*}}] = std::move(
  %result = tensor.insert %ct into %empty[%c0] : tensor<1x!cheddar.ciphertext>
  return %result : tensor<1x!cheddar.ciphertext>
}

// Test tensor.extract_slice
// CHECK: {{.*}} test_extract_slice
func.func @test_extract_slice(
    %ctx: !cheddar.context,
    %t: tensor<8x!cheddar.ciphertext>) -> tensor<4x!cheddar.ciphertext> {
  // CHECK: std::vector<Ct> {{.*}}({{.*}}.begin() + 2, {{.*}}.begin() + 2 + 4);
  %result = tensor.extract_slice %t[2] [4] [1] : tensor<8x!cheddar.ciphertext> to tensor<4x!cheddar.ciphertext>
  return %result : tensor<4x!cheddar.ciphertext>
}

// Numeric tensor.insert into tensor.empty should allocate a fresh zeroed result
// instead of copying the empty destination vector.
// CHECK: {{.*}} test_insert_numeric_into_empty
func.func @test_insert_numeric_into_empty(%x: f32) -> tensor<4xf32> {
  %empty = tensor.empty() : tensor<4xf32>
  %c1 = arith.constant 1 : index
  // CHECK: std::vector<double> [[EMPTY:.*]](4);
  // CHECK: std::vector<double> [[RES:.*]](4);
  // CHECK-NOT: [[RES]] = [[EMPTY]]
  // CHECK: [[RES]][{{.*}}] = {{.*}};
  %result = tensor.insert %x into %empty[%c1] : tensor<4xf32>
  return %result : tensor<4xf32>
}

// Numeric tensor.insert_slice into tensor.empty should also allocate a fresh
// zeroed result instead of copying the empty destination.
// CHECK: {{.*}} test_insert_slice_numeric_into_empty
func.func @test_insert_slice_numeric_into_empty(%src: tensor<2xf32>) -> tensor<4xf32> {
  %empty = tensor.empty() : tensor<4xf32>
  // CHECK: std::vector<double> [[EMPTY:.*]](4);
  // CHECK: std::vector<double> [[RES:.*]](4);
  // CHECK-NOT: [[RES]] = [[EMPTY]]
  // CHECK: std::copy({{.*}}.begin(), {{.*}}.end(), [[RES]].begin() + 1);
  %result = tensor.insert_slice %src into %empty[1] [2] [1] : tensor<2xf32> into tensor<4xf32>
  return %result : tensor<4xf32>
}
