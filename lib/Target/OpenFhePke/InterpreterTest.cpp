#include <string>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Target/OpenFhePke/Interpreter.h"
#include "mlir/include/mlir/Parser/Parser.h"  // from @llvm-project
#include "src/pke/include/openfhe.h"          // from @openfhe

namespace mlir {
namespace heir {
namespace openfhe {

using namespace lbcrypto;

OwningOpRef<ModuleOp> parseTest(MLIRContext* context,
                                const std::string& mlirStr) {
  return parseSourceString<ModuleOp>(mlirStr, context);
}

TEST(InterpreterTest, TestTrivial) {
  MLIRContext context;
  initContext(context);
  auto module = parseTest(&context, R"mlir(
    module {
      func.func @main() {
        return
      }
    }
  )mlir");
  Interpreter interpreter(module.get());
  std::string entryFunction = "main";
  std::vector<TypedCppValue> results = interpreter.interpret(entryFunction, {});
  EXPECT_TRUE(results.empty());
}

TEST(InterpreterTest, TestAdd) {
  MLIRContext context;
  initContext(context);
  auto module = parseTest(&context, R"mlir(
    module {
      func.func @main(%a: i32, %b: i32) -> i32 {
        %c = arith.addi %a, %b : i32
        return %c : i32
      }
    }
  )mlir");
  Interpreter interpreter(module.get());
  std::string entryFunction = "main";
  std::vector<TypedCppValue> inputs = {TypedCppValue(3), TypedCppValue(4)};
  std::vector<TypedCppValue> results =
      interpreter.interpret(entryFunction, inputs);
  EXPECT_EQ(results.size(), 1);
  EXPECT_EQ(std::get<int>(results[0].value), 7);
}

TEST(InterpreterTest, TestElementwiseAdd) {
  MLIRContext context;
  initContext(context);
  auto module = parseTest(&context, R"mlir(
    module {
      func.func @main(%a: tensor<3xi32>, %b: tensor<3xi32>) -> tensor<3xi32> {
        %c = arith.addi %a, %b : tensor<3xi32>
        return %c : tensor<3xi32>
      }
    }
  )mlir");
  Interpreter interpreter(module.get());
  std::string entryFunction = "main";
  std::vector<int> a = {1, 2, 3};
  std::vector<int> b = {2, 3, 4};
  std::vector<int> expected = {3, 5, 7};
  std::vector<TypedCppValue> inputs = {TypedCppValue(a), TypedCppValue(b)};
  std::vector<TypedCppValue> results =
      interpreter.interpret(entryFunction, inputs);
  EXPECT_EQ(results.size(), 1);
  EXPECT_EQ(std::get<std::vector<int>>(results[0].value), expected);
}

TEST(InterpreterTest, TestMul) {
  MLIRContext context;
  initContext(context);
  auto module = parseTest(&context, R"mlir(
    module {
      func.func @main(%a: i32, %b: i32) -> i32 {
        %c = arith.muli %a, %b : i32
        return %c : i32
      }
    }
  )mlir");
  Interpreter interpreter(module.get());
  std::string entryFunction = "main";
  std::vector<TypedCppValue> inputs = {TypedCppValue(3), TypedCppValue(4)};
  std::vector<TypedCppValue> results =
      interpreter.interpret(entryFunction, inputs);
  EXPECT_EQ(results.size(), 1);
  EXPECT_EQ(std::get<int>(results[0].value), 12);
}

TEST(InterpreterTest, TestDiv) {
  MLIRContext context;
  initContext(context);
  auto module = parseTest(&context, R"mlir(
    module {
      func.func @main(%a: i32, %b: i32) -> i32 {
        %c = arith.divsi %a, %b : i32
        return %c : i32
      }
    }
  )mlir");
  Interpreter interpreter(module.get());
  std::string entryFunction = "main";
  std::vector<TypedCppValue> inputs = {TypedCppValue(12), TypedCppValue(3)};
  std::vector<TypedCppValue> results =
      interpreter.interpret(entryFunction, inputs);
  EXPECT_EQ(results.size(), 1);
  EXPECT_EQ(std::get<int>(results[0].value), 4);
}

TEST(InterpreterTest, TestRem) {
  MLIRContext context;
  initContext(context);
  auto module = parseTest(&context, R"mlir(
    module {
      func.func @main(%a: i32, %b: i32) -> i32 {
        %c = arith.remsi %a, %b : i32
        return %c : i32
      }
    }
  )mlir");
  Interpreter interpreter(module.get());
  std::string entryFunction = "main";
  std::vector<TypedCppValue> inputs = {TypedCppValue(10), TypedCppValue(3)};
  std::vector<TypedCppValue> results =
      interpreter.interpret(entryFunction, inputs);
  EXPECT_EQ(results.size(), 1);
  EXPECT_EQ(std::get<int>(results[0].value), 1);
}

TEST(InterpreterTest, TestAnd) {
  MLIRContext context;
  initContext(context);
  auto module = parseTest(&context, R"mlir(
    module {
      func.func @main(%a: i32, %b: i32) -> i32 {
        %c = arith.andi %a, %b : i32
        return %c : i32
      }
    }
  )mlir");
  Interpreter interpreter(module.get());
  std::string entryFunction = "main";
  std::vector<TypedCppValue> inputs = {TypedCppValue(1), TypedCppValue(1)};
  std::vector<TypedCppValue> results =
      interpreter.interpret(entryFunction, inputs);
  EXPECT_EQ(results.size(), 1);
  EXPECT_EQ(std::get<int>(results[0].value), 1);
}

TEST(InterpreterTest, TestCmpILt) {
  MLIRContext context;
  initContext(context);
  auto module = parseTest(&context, R"mlir(
    module {
      func.func @main(%a: i32, %b: i32) -> i32 {
        %cmp = arith.cmpi slt, %a, %b : i32
        %c = arith.extui %cmp : i1 to i32
        return %c : i32
      }
    }
  )mlir");
  Interpreter interpreter(module.get());
  std::string entryFunction = "main";
  std::vector<TypedCppValue> inputs = {TypedCppValue(3), TypedCppValue(5)};
  std::vector<TypedCppValue> results =
      interpreter.interpret(entryFunction, inputs);
  EXPECT_EQ(results.size(), 1);
  EXPECT_EQ(std::get<int>(results[0].value), 1);
}

TEST(InterpreterTest, TestCmpIEq) {
  MLIRContext context;
  initContext(context);
  auto module = parseTest(&context, R"mlir(
    module {
      func.func @main(%a: i32, %b: i32) -> i32 {
        %cmp = arith.cmpi eq, %a, %b : i32
        %c = arith.extui %cmp : i1 to i32
        return %c : i32
      }
    }
  )mlir");
  Interpreter interpreter(module.get());
  std::string entryFunction = "main";
  std::vector<TypedCppValue> inputs = {TypedCppValue(5), TypedCppValue(5)};
  std::vector<TypedCppValue> results =
      interpreter.interpret(entryFunction, inputs);
  EXPECT_EQ(results.size(), 1);
  EXPECT_EQ(std::get<int>(results[0].value), 1);
}

TEST(InterpreterTest, TestSelect) {
  MLIRContext context;
  initContext(context);
  auto module = parseTest(&context, R"mlir(
    module {
      func.func @main(%a: i32, %b: i32) -> i32 {
        %cmp = arith.cmpi slt, %a, %b : i32
        %c = arith.select %cmp, %a, %b : i32
        return %c : i32
      }
    }
  )mlir");
  Interpreter interpreter(module.get());
  std::string entryFunction = "main";
  std::vector<TypedCppValue> inputs = {TypedCppValue(10), TypedCppValue(20)};
  std::vector<TypedCppValue> results =
      interpreter.interpret(entryFunction, inputs);
  EXPECT_EQ(results.size(), 1);
  EXPECT_EQ(std::get<int>(results[0].value), 10);
}

TEST(InterpreterTest, TestSelectFalse) {
  MLIRContext context;
  initContext(context);
  auto module = parseTest(&context, R"mlir(
    module {
      func.func @main(%a: i32, %b: i32) -> i32 {
        %cmp = arith.cmpi slt, %a, %b : i32
        %c = arith.select %cmp, %a, %b : i32
        return %c : i32
      }
    }
  )mlir");
  Interpreter interpreter(module.get());
  std::string entryFunction = "main";
  std::vector<TypedCppValue> inputs = {TypedCppValue(20), TypedCppValue(10)};
  std::vector<TypedCppValue> results =
      interpreter.interpret(entryFunction, inputs);
  EXPECT_EQ(results.size(), 1);
  EXPECT_EQ(std::get<int>(results[0].value), 10);
}

TEST(InterpreterTest, TestTensorSplat) {
  MLIRContext context;
  initContext(context);
  auto module = parseTest(&context, R"mlir(
    module {
      func.func @main(%val: i32) -> tensor<4xi32> {
        %t = tensor.splat %val : tensor<4xi32>
        return %t : tensor<4xi32>
      }
    }
  )mlir");
  Interpreter interpreter(module.get());
  std::string entryFunction = "main";
  std::vector<TypedCppValue> inputs = {TypedCppValue(42)};
  std::vector<TypedCppValue> results =
      interpreter.interpret(entryFunction, inputs);
  EXPECT_EQ(results.size(), 1);
  auto vec = std::get<std::vector<int>>(results[0].value);
  EXPECT_EQ(vec.size(), 4);
  EXPECT_EQ(vec[0], 42);
  EXPECT_EQ(vec[1], 42);
  EXPECT_EQ(vec[2], 42);
  EXPECT_EQ(vec[3], 42);
}

TEST(InterpreterTest, TestTensorFromElements) {
  MLIRContext context;
  initContext(context);
  auto module = parseTest(&context, R"mlir(
    module {
      func.func @main(%a: i32, %b: i32, %c: i32) -> tensor<3xi32> {
        %t = tensor.from_elements %a, %b, %c : tensor<3xi32>
        return %t : tensor<3xi32>
      }
    }
  )mlir");
  Interpreter interpreter(module.get());
  std::string entryFunction = "main";
  std::vector<TypedCppValue> inputs = {TypedCppValue(10), TypedCppValue(20),
                                       TypedCppValue(30)};
  std::vector<TypedCppValue> results =
      interpreter.interpret(entryFunction, inputs);
  EXPECT_EQ(results.size(), 1);
  auto vec = std::get<std::vector<int>>(results[0].value);
  EXPECT_EQ(vec.size(), 3);
  EXPECT_EQ(vec[0], 10);
  EXPECT_EQ(vec[1], 20);
  EXPECT_EQ(vec[2], 30);
}

TEST(InterpreterTest, TestTensorExtract) {
  MLIRContext context;
  initContext(context);
  auto module = parseTest(&context, R"mlir(
    module {
      func.func @main(%a: i32, %b: i32, %c: i32) -> i32 {
        %t = tensor.from_elements %a, %b, %c : tensor<3xi32>
        %idx = arith.constant 1 : index
        %val = tensor.extract %t[%idx] : tensor<3xi32>
        return %val : i32
      }
    }
  )mlir");
  Interpreter interpreter(module.get());
  std::string entryFunction = "main";
  std::vector<TypedCppValue> inputs = {TypedCppValue(10), TypedCppValue(20),
                                       TypedCppValue(30)};
  std::vector<TypedCppValue> results =
      interpreter.interpret(entryFunction, inputs);
  EXPECT_EQ(results.size(), 1);
  EXPECT_EQ(std::get<int>(results[0].value), 20);
}

TEST(InterpreterTest, TestTensorInsert) {
  MLIRContext context;
  initContext(context);
  auto module = parseTest(&context, R"mlir(
    module {
      func.func @main(%val: i32) -> tensor<3xi32> {
        %t = tensor.empty() : tensor<3xi32>
        %idx = arith.constant 1 : index
        %t2 = tensor.insert %val into %t[%idx] : tensor<3xi32>
        return %t2 : tensor<3xi32>
      }
    }
  )mlir");
  Interpreter interpreter(module.get());
  std::string entryFunction = "main";
  std::vector<TypedCppValue> inputs = {TypedCppValue(99)};
  std::vector<TypedCppValue> results =
      interpreter.interpret(entryFunction, inputs);
  EXPECT_EQ(results.size(), 1);
  auto vec = std::get<std::vector<int>>(results[0].value);
  EXPECT_EQ(vec.size(), 3);
  EXPECT_EQ(vec[1], 99);
}

// Helper function to set up a basic BGV crypto context for testing
struct CryptoSetup {
  CryptoContext<DCRTPoly> cc;
  KeyPair<DCRTPoly> keyPair;

  CryptoSetup(uint32_t multDepth = 2) {
    CCParams<CryptoContextBGVRNS> parameters;
    parameters.SetPlaintextModulus(65537);
    parameters.SetMultiplicativeDepth(multDepth);

    cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);
  }
};

// Common LWE type definitions header for MLIR tests
static const char* kLWETypesHeader = R"mlir(
!Z65537 = !mod_arith.int<65537 : i64>
!Z1095233372161 = !mod_arith.int<1095233372161 : i64>
!rns_L0 = !rns.rns<!Z1095233372161>
#ring_pt = #polynomial.ring<coefficientType = !Z65537, polynomialModulus = <1 + x**32>>
#ring_ct = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**32>>
#encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain = #lwe.modulus_chain<elements = <1095233372161 : i64>, current = 0>
#plaintext_space = #lwe.plaintext_space<ring = #ring_pt, encoding = #encoding>
#ciphertext_space = #lwe.ciphertext_space<ring = #ring_ct, encryption_type = lsb>
!ct = !lwe.lwe_ciphertext<application_data = <message_type = i32>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space, key = #key, modulus_chain = #modulus_chain>
!pt = !lwe.lwe_plaintext<application_data = <message_type = i32>, plaintext_space = #plaintext_space>
)mlir";

TEST(InterpreterTest, TestOpenfheAdd) {
  CryptoSetup setup;

  // Create plaintexts
  std::vector<int64_t> vec1 = {1, 2, 3, 4};
  std::vector<int64_t> vec2 = {5, 6, 7, 8};
  auto pt1 = setup.cc->MakePackedPlaintext(vec1);
  auto pt2 = setup.cc->MakePackedPlaintext(vec2);

  // Encrypt
  auto ct1 = setup.cc->Encrypt(setup.keyPair.publicKey, pt1);
  auto ct2 = setup.cc->Encrypt(setup.keyPair.publicKey, pt2);

  // Test via interpreter
  MLIRContext context;
  initContext(context);
  std::string mlirStr = std::string(kLWETypesHeader) + R"mlir(
module {
  func.func @main(%cc: !openfhe.crypto_context, %ct1: !ct, %ct2: !ct) -> !ct {
    %result = openfhe.add %cc, %ct1, %ct2 : (!openfhe.crypto_context, !ct, !ct) -> !ct
    return %result : !ct
  }
}
)mlir";
  auto module = parse(&context, mlirStr);

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {TypedCppValue(setup.cc),
                                       TypedCppValue(ct1), TypedCppValue(ct2)};
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);

  EXPECT_EQ(results.size(), 1);
  auto resultCt = std::get<CiphertextT>(results[0].value);

  // Decrypt and verify
  Plaintext resultPt;
  setup.cc->Decrypt(setup.keyPair.secretKey, resultCt, &resultPt);
  resultPt->SetLength(vec1.size());

  auto resultVec = resultPt->GetPackedValue();
  EXPECT_EQ(resultVec.size(), vec1.size());
  for (size_t i = 0; i < vec1.size(); i++) {
    EXPECT_EQ(resultVec[i], vec1[i] + vec2[i]);
  }
}

TEST(InterpreterTest, TestOpenfheSub) {
  CryptoSetup setup;

  std::vector<int64_t> vec1 = {10, 20, 30, 40};
  std::vector<int64_t> vec2 = {3, 5, 7, 9};
  auto pt1 = setup.cc->MakePackedPlaintext(vec1);
  auto pt2 = setup.cc->MakePackedPlaintext(vec2);

  auto ct1 = setup.cc->Encrypt(setup.keyPair.publicKey, pt1);
  auto ct2 = setup.cc->Encrypt(setup.keyPair.publicKey, pt2);

  MLIRContext context;
  initContext(context);
  std::string mlirStr = std::string(kLWETypesHeader) + R"mlir(
module {
      func.func @main(%cc: !openfhe.crypto_context, %ct1: !ct, %ct2: !ct) -> !ct {
        %result = openfhe.sub %cc, %ct1, %ct2 : (!openfhe.crypto_context, !ct, !ct) -> !ct
        return %result : !ct
      }
}
)mlir";
  auto module = parse(&context, mlirStr);

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {TypedCppValue(setup.cc),
                                       TypedCppValue(ct1), TypedCppValue(ct2)};
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);

  auto resultCt = std::get<CiphertextT>(results[0].value);
  Plaintext resultPt;
  setup.cc->Decrypt(setup.keyPair.secretKey, resultCt, &resultPt);
  resultPt->SetLength(vec1.size());

  auto resultVec = resultPt->GetPackedValue();
  for (size_t i = 0; i < vec1.size(); i++) {
    EXPECT_EQ(resultVec[i], vec1[i] - vec2[i]);
  }
}

TEST(InterpreterTest, TestOpenfheMul) {
  CryptoSetup setup(2);  // Need depth 2 for multiplication

  std::vector<int64_t> vec1 = {2, 3, 4, 5};
  std::vector<int64_t> vec2 = {3, 4, 5, 6};
  auto pt1 = setup.cc->MakePackedPlaintext(vec1);
  auto pt2 = setup.cc->MakePackedPlaintext(vec2);

  auto ct1 = setup.cc->Encrypt(setup.keyPair.publicKey, pt1);
  auto ct2 = setup.cc->Encrypt(setup.keyPair.publicKey, pt2);

  MLIRContext context;
  initContext(context);
  std::string mlirStr = std::string(kLWETypesHeader) + R"mlir(
module {
      func.func @main(%cc: !openfhe.crypto_context, %ct1: !ct, %ct2: !ct) -> !ct {
        %result = openfhe.mul %cc, %ct1, %ct2 : (!openfhe.crypto_context, !ct, !ct) -> !ct
        return %result : !ct
      }
}
)mlir";
  auto module = parse(&context, mlirStr);

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {TypedCppValue(setup.cc),
                                       TypedCppValue(ct1), TypedCppValue(ct2)};
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);

  auto resultCt = std::get<CiphertextT>(results[0].value);
  Plaintext resultPt;
  setup.cc->Decrypt(setup.keyPair.secretKey, resultCt, &resultPt);
  resultPt->SetLength(vec1.size());

  auto resultVec = resultPt->GetPackedValue();
  for (size_t i = 0; i < vec1.size(); i++) {
    EXPECT_EQ(resultVec[i], vec1[i] * vec2[i]);
  }
}

TEST(InterpreterTest, TestOpenfheMulPlain) {
  CryptoSetup setup;

  std::vector<int64_t> vec1 = {2, 3, 4, 5};
  std::vector<int64_t> vec2 = {10, 10, 10, 10};
  auto pt1 = setup.cc->MakePackedPlaintext(vec1);
  auto pt2 = setup.cc->MakePackedPlaintext(vec2);

  auto ct1 = setup.cc->Encrypt(setup.keyPair.publicKey, pt1);

  MLIRContext context;
  initContext(context);
  std::string mlirStr = std::string(kLWETypesHeader) + R"mlir(
module {
      func.func @main(%cc: !openfhe.crypto_context, %ct: !ct, %pt: !pt) -> !ct {
        %result = openfhe.mul_plain %cc, %ct, %pt : (!openfhe.crypto_context, !ct, !pt) -> !ct
        return %result : !ct
      }
}
)mlir";
  auto module = parse(&context, mlirStr);

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {TypedCppValue(setup.cc),
                                       TypedCppValue(ct1), TypedCppValue(pt2)};
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);

  auto resultCt = std::get<CiphertextT>(results[0].value);
  Plaintext resultPt;
  setup.cc->Decrypt(setup.keyPair.secretKey, resultCt, &resultPt);
  resultPt->SetLength(vec1.size());

  auto resultVec = resultPt->GetPackedValue();
  for (size_t i = 0; i < vec1.size(); i++) {
    EXPECT_EQ(resultVec[i], vec1[i] * vec2[i]);
  }
}

TEST(InterpreterTest, TestOpenfheNegate) {
  CryptoSetup setup;

  std::vector<int64_t> vec = {5, 10, 15, 20};
  auto pt = setup.cc->MakePackedPlaintext(vec);
  auto ct = setup.cc->Encrypt(setup.keyPair.publicKey, pt);

  MLIRContext context;
  initContext(context);
  std::string mlirStr = std::string(kLWETypesHeader) + R"mlir(
module {
      func.func @main(%cc: !openfhe.crypto_context, %ct: !ct) -> !ct {
        %result = openfhe.negate %cc, %ct : (!openfhe.crypto_context, !ct) -> !ct
        return %result : !ct
      }
}
)mlir";
  auto module = parse(&context, mlirStr);

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {TypedCppValue(setup.cc),
                                       TypedCppValue(ct)};
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);

  auto resultCt = std::get<CiphertextT>(results[0].value);
  Plaintext resultPt;
  setup.cc->Decrypt(setup.keyPair.secretKey, resultCt, &resultPt);
  resultPt->SetLength(vec.size());

  auto resultVec = resultPt->GetPackedValue();
  for (size_t i = 0; i < vec.size(); i++) {
    EXPECT_EQ(resultVec[i], -vec[i]);
  }
}

TEST(InterpreterTest, TestOpenfheSquare) {
  CryptoSetup setup(2);  // Need depth for squaring

  std::vector<int64_t> vec = {2, 3, 4, 5};
  auto pt = setup.cc->MakePackedPlaintext(vec);
  auto ct = setup.cc->Encrypt(setup.keyPair.publicKey, pt);

  MLIRContext context;
  initContext(context);
  std::string mlirStr = std::string(kLWETypesHeader) + R"mlir(
module {
      func.func @main(%cc: !openfhe.crypto_context, %ct: !ct) -> !ct {
        %result = openfhe.square %cc, %ct : (!openfhe.crypto_context, !ct) -> !ct
        return %result : !ct
      }
}
)mlir";
  auto module = parse(&context, mlirStr);

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {TypedCppValue(setup.cc),
                                       TypedCppValue(ct)};
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);

  auto resultCt = std::get<CiphertextT>(results[0].value);
  Plaintext resultPt;
  setup.cc->Decrypt(setup.keyPair.secretKey, resultCt, &resultPt);
  resultPt->SetLength(vec.size());

  auto resultVec = resultPt->GetPackedValue();
  for (size_t i = 0; i < vec.size(); i++) {
    EXPECT_EQ(resultVec[i], vec[i] * vec[i]);
  }
}

TEST(InterpreterTest, TestOpenfheRot) {
  CryptoSetup setup;

  // Generate rotation keys
  setup.cc->EvalRotateKeyGen(setup.keyPair.secretKey, {2});

  std::vector<int64_t> vec = {1, 2, 3, 4, 5, 6, 7, 8};
  // Cyclically replicate the vector to fill the slots
  std::vector<int64_t> replicatedVec;
  int64_t numSlots = setup.cc->GetRingDimension() / 2;
  for (int i = 0; i < numSlots; i++)
    replicatedVec.push_back(vec[i % vec.size()]);

  auto pt = setup.cc->MakePackedPlaintext(replicatedVec);
  auto ct = setup.cc->Encrypt(setup.keyPair.publicKey, pt);

  MLIRContext context;
  initContext(context);
  std::string mlirStr = std::string(kLWETypesHeader) + R"mlir(
module {
      func.func @main(%cc: !openfhe.crypto_context, %ct: !ct) -> !ct {
        %result = openfhe.rot %cc, %ct {index = 2 : i32} : (!openfhe.crypto_context, !ct) -> !ct
        return %result : !ct
      }
}
)mlir";
  auto module = parse(&context, mlirStr);

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {TypedCppValue(setup.cc),
                                       TypedCppValue(ct)};
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);

  auto resultCt = std::get<CiphertextT>(results[0].value);
  Plaintext resultPt;
  setup.cc->Decrypt(setup.keyPair.secretKey, resultCt, &resultPt);
  resultPt->SetLength(vec.size());

  auto resultVec = resultPt->GetPackedValue();
  // Rotation by 2 should shift: [1,2,3,4,5,6,7,8] -> [3,4,5,6,7,8,1,2]
  EXPECT_EQ(resultVec[0], 3);
  EXPECT_EQ(resultVec[1], 4);
  EXPECT_EQ(resultVec[6], 1);
  EXPECT_EQ(resultVec[7], 2);
}

TEST(InterpreterTest, TestOpenfheMakePackedPlaintext) {
  CryptoSetup setup;

  MLIRContext context;
  initContext(context);
  std::string mlirStr = std::string(kLWETypesHeader) + R"mlir(
module {
      func.func @main(%cc: !openfhe.crypto_context, %vec: tensor<4xi32>) -> !pt {
        %result = openfhe.make_packed_plaintext %cc, %vec : (!openfhe.crypto_context, tensor<4xi32>) -> !pt
        return %result : !pt
      }
}
)mlir";
  auto module = parseTest(&context, mlirStr);

  Interpreter interpreter(module.get());

  std::vector<int> vec = {10, 20, 30, 40};
  std::vector<TypedCppValue> inputs = {TypedCppValue(setup.cc),
                                       TypedCppValue(vec)};
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);

  EXPECT_EQ(results.size(), 1);
  auto resultPt = std::get<PlaintextT>(results[0].value);
  resultPt->SetLength(vec.size());

  auto resultVec = resultPt->GetPackedValue();
  EXPECT_EQ(resultVec.size(), vec.size());
  for (size_t i = 0; i < vec.size(); i++) {
    EXPECT_EQ(resultVec[i], vec[i]);
  }
}

TEST(InterpreterTest, TestOpenfheEncryptDecrypt) {
  CryptoSetup setup;

  std::vector<int64_t> vec = {7, 14, 21, 28};
  auto pt = setup.cc->MakePackedPlaintext(vec);

  MLIRContext context;
  initContext(context);
  std::string mlirStr = std::string(kLWETypesHeader) + R"mlir(
module {
      func.func @main(%cc: !openfhe.crypto_context, %pt: !pt, %pk: !openfhe.public_key, %sk: !openfhe.private_key) -> !pt {
        %ct = openfhe.encrypt %cc, %pt, %pk : (!openfhe.crypto_context, !pt, !openfhe.public_key) -> !ct
        %result = openfhe.decrypt %cc, %ct, %sk : (!openfhe.crypto_context, !ct, !openfhe.private_key) -> !pt
        return %result : !pt
      }
}
)mlir";
  auto module = parse(&context, mlirStr);

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {TypedCppValue(setup.cc),
                                       TypedCppValue(pt),
                                       TypedCppValue(setup.keyPair.publicKey),
                                       TypedCppValue(setup.keyPair.secretKey)};
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);

  auto resultPt = std::get<PlaintextT>(results[0].value);
  resultPt->SetLength(vec.size());

  auto resultVec = resultPt->GetPackedValue();
  for (size_t i = 0; i < vec.size(); i++) {
    EXPECT_EQ(resultVec[i], vec[i]);
  }
}

TEST(InterpreterTest, TestOpenfheRLWEDecodeBGVScalar) {
  CryptoSetup setup;

  std::vector<int64_t> vec = {42};
  auto pt = setup.cc->MakePackedPlaintext(vec);

  MLIRContext context;
  initContext(context);
  std::string mlirStr = std::string(kLWETypesHeader) + R"mlir(
module attributes {scheme.bgv} {
  func.func @main(%pt: !pt) -> i32 {
    %result = lwe.rlwe_decode %pt {encoding = #encoding, ring = #ring_pt} : !pt -> i32
    return %result : i32
  }
}
)mlir";
  auto module = parse(&context, mlirStr);

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {TypedCppValue(pt)};
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);

  EXPECT_EQ(results.size(), 1);
  EXPECT_EQ(std::get<int>(results[0].value), 42);
}

TEST(InterpreterTest, TestOpenfheRLWEDecodeBGVTensor) {
  CryptoSetup setup;

  std::vector<int64_t> vec = {1, 2, 3, 4, 5, 6, 7, 8};
  auto pt = setup.cc->MakePackedPlaintext(vec);

  MLIRContext context;
  initContext(context);
  std::string mlirStr = std::string(kLWETypesHeader) + R"mlir(
module attributes {scheme.bgv} {
  func.func @main(%pt: !pt) -> tensor<8xi32> {
    %result = lwe.rlwe_decode %pt {encoding = #encoding, ring = #ring_pt} : !pt -> tensor<8xi32>
    return %result : tensor<8xi32>
  }
}
)mlir";
  auto module = parse(&context, mlirStr);

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {TypedCppValue(pt)};
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);

  EXPECT_EQ(results.size(), 1);
  auto resultVec = std::get<std::vector<int>>(results[0].value);
  EXPECT_EQ(resultVec.size(), vec.size());
  for (size_t i = 0; i < vec.size(); i++) {
    EXPECT_EQ(resultVec[i], vec[i]);
  }
}

// Helper function to set up a basic CKKS crypto context for testing
struct CKKSSetup {
  CryptoContext<DCRTPoly> cc;
  KeyPair<DCRTPoly> keyPair;

  CKKSSetup(uint32_t multDepth = 2) {
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(50);
    parameters.SetBatchSize(8);

    cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);
  }
};

TEST(InterpreterTest, TestOpenfheRLWEDecodeCKKSScalar) {
  CKKSSetup setup;

  std::vector<double> vec = {3.14};
  auto pt = setup.cc->MakeCKKSPackedPlaintext(vec);

  MLIRContext context;
  initContext(context);
  std::string mlirStr = R"mlir(
!Z65537 = !mod_arith.int<65537 : i64>
!Z1095233372161 = !mod_arith.int<1095233372161 : i64>
!rns_L0 = !rns.rns<!Z1095233372161>
#ring_pt = #polynomial.ring<coefficientType = !Z65537, polynomialModulus = <1 + x**32>>
#ckks_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain = #lwe.modulus_chain<elements = <1095233372161 : i64>, current = 0>
#plaintext_space = #lwe.plaintext_space<ring = #ring_pt, encoding = #ckks_encoding>
!pt = !lwe.lwe_plaintext<application_data = <message_type = f32>, plaintext_space = #plaintext_space>

module attributes {scheme.ckks} {
  func.func @main(%pt: !pt) -> f32 {
    %result = lwe.rlwe_decode %pt {encoding = #ckks_encoding, ring = #ring_pt} : !pt -> f32
    return %result : f32
  }
}
)mlir";
  auto module = parse(&context, mlirStr);

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {TypedCppValue(pt)};
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);

  EXPECT_EQ(results.size(), 1);
  // Use approximate comparison for floating point
  EXPECT_NEAR(std::get<float>(results[0].value), 3.14f, 0.01f);
}

TEST(InterpreterTest, TestOpenfheRLWEDecodeCKKSTensor) {
  CKKSSetup setup;

  std::vector<double> vec = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8};
  auto pt = setup.cc->MakeCKKSPackedPlaintext(vec);

  MLIRContext context;
  initContext(context);
  std::string mlirStr = R"mlir(
!Z65537 = !mod_arith.int<65537 : i64>
!Z1095233372161 = !mod_arith.int<1095233372161 : i64>
!rns_L0 = !rns.rns<!Z1095233372161>
#ring_pt = #polynomial.ring<coefficientType = !Z65537, polynomialModulus = <1 + x**32>>
#ckks_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain = #lwe.modulus_chain<elements = <1095233372161 : i64>, current = 0>
#plaintext_space = #lwe.plaintext_space<ring = #ring_pt, encoding = #ckks_encoding>
!pt = !lwe.lwe_plaintext<application_data = <message_type = tensor<8xf32>>, plaintext_space = #plaintext_space>

module attributes {scheme.ckks} {
  func.func @main(%pt: !pt) -> tensor<8xf32> {
    %result = lwe.rlwe_decode %pt {encoding = #ckks_encoding, ring = #ring_pt} : !pt -> tensor<8xf32>
    return %result : tensor<8xf32>
  }
}
)mlir";
  auto module = parse(&context, mlirStr);

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {TypedCppValue(pt)};
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);

  EXPECT_EQ(results.size(), 1);
  auto resultVec = std::get<std::vector<float>>(results[0].value);
  EXPECT_EQ(resultVec.size(), vec.size());
  for (size_t i = 0; i < vec.size(); i++) {
    EXPECT_NEAR(resultVec[i], static_cast<float>(vec[i]), 0.01f);
  }
}

TEST(InterpreterTest, TestDoubleConstant) {
  MLIRContext context;
  initContext(context);
  auto module = parseTest(&context, R"mlir(
    module {
      func.func @main() -> f64 {
        %c = arith.constant 3.14159265358979 : f64
        return %c : f64
      }
    }
  )mlir");
  Interpreter interpreter(module.get());
  std::string entryFunction = "main";
  std::vector<TypedCppValue> results = interpreter.interpret(entryFunction, {});
  EXPECT_EQ(results.size(), 1);
  EXPECT_NEAR(std::get<double>(results[0].value), 3.14159265358979, 1e-10);
}

TEST(InterpreterTest, TestFloatConstant) {
  MLIRContext context;
  initContext(context);
  auto module = parseTest(&context, R"mlir(
    module {
      func.func @main() -> f32 {
        %c = arith.constant 3.14 : f32
        return %c : f32
      }
    }
  )mlir");
  Interpreter interpreter(module.get());
  std::string entryFunction = "main";
  std::vector<TypedCppValue> results = interpreter.interpret(entryFunction, {});
  EXPECT_EQ(results.size(), 1);
  EXPECT_NEAR(std::get<float>(results[0].value), 3.14f, 0.01f);
}

TEST(InterpreterTest, TestDoubleTensorConstant) {
  MLIRContext context;
  initContext(context);
  auto module = parseTest(&context, R"mlir(
    module {
      func.func @main() -> tensor<3xf64> {
        %c = arith.constant dense<[1.1, 2.2, 3.3]> : tensor<3xf64>
        return %c : tensor<3xf64>
      }
    }
  )mlir");
  Interpreter interpreter(module.get());
  std::string entryFunction = "main";
  std::vector<TypedCppValue> results = interpreter.interpret(entryFunction, {});
  EXPECT_EQ(results.size(), 1);
  auto vec = std::get<std::vector<double>>(results[0].value);
  EXPECT_EQ(vec.size(), 3);
  EXPECT_NEAR(vec[0], 1.1, 0.01);
  EXPECT_NEAR(vec[1], 2.2, 0.01);
  EXPECT_NEAR(vec[2], 3.3, 0.01);
}

TEST(InterpreterTest, TestDoubleTensorSplat) {
  MLIRContext context;
  initContext(context);
  auto module = parseTest(&context, R"mlir(
    module {
      func.func @main(%val: f64) -> tensor<4xf64> {
        %t = tensor.splat %val : tensor<4xf64>
        return %t : tensor<4xf64>
      }
    }
  )mlir");
  Interpreter interpreter(module.get());
  std::string entryFunction = "main";
  std::vector<TypedCppValue> inputs = {TypedCppValue(3.14)};
  std::vector<TypedCppValue> results =
      interpreter.interpret(entryFunction, inputs);
  EXPECT_EQ(results.size(), 1);
  auto vec = std::get<std::vector<double>>(results[0].value);
  EXPECT_EQ(vec.size(), 4);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_NEAR(vec[i], 3.14, 0.01);
  }
}

TEST(InterpreterTest, TestExtFOpFloatToDouble) {
  MLIRContext context;
  initContext(context);
  auto module = parseTest(&context, R"mlir(
    module {
      func.func @main(%a: f32) -> f64 {
        %c = arith.extf %a : f32 to f64
        return %c : f64
      }
    }
  )mlir");
  Interpreter interpreter(module.get());
  std::string entryFunction = "main";
  std::vector<TypedCppValue> inputs = {TypedCppValue(3.14f)};
  std::vector<TypedCppValue> results =
      interpreter.interpret(entryFunction, inputs);
  EXPECT_EQ(results.size(), 1);
  EXPECT_NEAR(std::get<double>(results[0].value), 3.14, 0.01);
}

TEST(InterpreterTest, TestExtFOpFloatVectorToDoubleVector) {
  MLIRContext context;
  initContext(context);
  auto module = parseTest(&context, R"mlir(
    module {
      func.func @main(%a: tensor<3xf32>) -> tensor<3xf64> {
        %c = arith.extf %a : tensor<3xf32> to tensor<3xf64>
        return %c : tensor<3xf64>
      }
    }
  )mlir");
  Interpreter interpreter(module.get());
  std::string entryFunction = "main";
  std::vector<float> a = {1.1f, 2.2f, 3.3f};
  std::vector<TypedCppValue> inputs = {TypedCppValue(a)};
  std::vector<TypedCppValue> results =
      interpreter.interpret(entryFunction, inputs);
  EXPECT_EQ(results.size(), 1);
  auto vec = std::get<std::vector<double>>(results[0].value);
  EXPECT_EQ(vec.size(), 3);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_NEAR(vec[i], a[i], 0.01);
  }
}

TEST(InterpreterTest, TestOpenfheRLWEDecodeCKKSScalarDouble) {
  CKKSSetup setup;

  std::vector<double> vec = {3.14159265358979};
  auto pt = setup.cc->MakeCKKSPackedPlaintext(vec);

  MLIRContext context;
  initContext(context);
  std::string mlirStr = R"mlir(
!Z65537 = !mod_arith.int<65537 : i64>
!Z1095233372161 = !mod_arith.int<1095233372161 : i64>
!rns_L0 = !rns.rns<!Z1095233372161>
#ring_pt = #polynomial.ring<coefficientType = !Z65537, polynomialModulus = <1 + x**32>>
#ckks_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain = #lwe.modulus_chain<elements = <1095233372161 : i64>, current = 0>
#plaintext_space = #lwe.plaintext_space<ring = #ring_pt, encoding = #ckks_encoding>
!pt = !lwe.lwe_plaintext<application_data = <message_type = f64>, plaintext_space = #plaintext_space>

module attributes {scheme.ckks} {
  func.func @main(%pt: !pt) -> f64 {
    %result = lwe.rlwe_decode %pt {encoding = #ckks_encoding, ring = #ring_pt} : !pt -> f64
    return %result : f64
  }
}
)mlir";
  auto module = parse(&context, mlirStr);

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {TypedCppValue(pt)};
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);

  EXPECT_EQ(results.size(), 1);
  // Use approximate comparison for floating point
  EXPECT_NEAR(std::get<double>(results[0].value), 3.14159265358979, 0.01);
}

TEST(InterpreterTest, TestOpenfheRLWEDecodeCKKSTensorDouble) {
  CKKSSetup setup;

  std::vector<double> vec = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8};
  auto pt = setup.cc->MakeCKKSPackedPlaintext(vec);

  MLIRContext context;
  initContext(context);
  std::string mlirStr = R"mlir(
!Z65537 = !mod_arith.int<65537 : i64>
!Z1095233372161 = !mod_arith.int<1095233372161 : i64>
!rns_L0 = !rns.rns<!Z1095233372161>
#ring_pt = #polynomial.ring<coefficientType = !Z65537, polynomialModulus = <1 + x**32>>
#ckks_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain = #lwe.modulus_chain<elements = <1095233372161 : i64>, current = 0>
#plaintext_space = #lwe.plaintext_space<ring = #ring_pt, encoding = #ckks_encoding>
!pt = !lwe.lwe_plaintext<application_data = <message_type = tensor<8xf64>>, plaintext_space = #plaintext_space>

module attributes {scheme.ckks} {
  func.func @main(%pt: !pt) -> tensor<8xf64> {
    %result = lwe.rlwe_decode %pt {encoding = #ckks_encoding, ring = #ring_pt} : !pt -> tensor<8xf64>
    return %result : tensor<8xf64>
  }
}
)mlir";
  auto module = parse(&context, mlirStr);

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {TypedCppValue(pt)};
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);

  EXPECT_EQ(results.size(), 1);
  auto resultVec = std::get<std::vector<double>>(results[0].value);
  EXPECT_EQ(resultVec.size(), vec.size());
  for (size_t i = 0; i < vec.size(); i++) {
    EXPECT_NEAR(resultVec[i], vec[i], 0.01);
  }
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
