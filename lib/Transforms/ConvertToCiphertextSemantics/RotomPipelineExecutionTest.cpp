#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Dialect/Rotom/IR/RotomDialect.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/LayoutAssignment.h"
#include "lib/Dialect/Rotom/Transforms/MaterializeTensorExtLayout/MaterializeTensorExtLayout.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Target/OpenFhePke/Interpreter.h"
#include "lib/Transforms/ConvertToCiphertextSemantics/ConvertToCiphertextSemantics.h"
#include "lib/Utils/Layout/Utils.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OwningOpRef.h"            // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace {

using openfhe::Interpreter;
using openfhe::TypedCppValue;

void initContext(MLIRContext& context) {
  openfhe::initContext(context);
  context.loadDialect<rotom::RotomDialect, secret::SecretDialect>();
}

LogicalResult runRotomPipeline(ModuleOp module, MLIRContext* context,
                               int ciphertextSize) {
  PassManager pm(context);
  pm.addPass(rotom::createLayoutAssignment());
  pm.addPass(rotom::createMaterializeTensorExtLayout());

  ConvertToCiphertextSemanticsOptions options;
  options.ciphertextSize = ciphertextSize;
  pm.addPass(createConvertToCiphertextSemantics(options));
  // Keep tensor_ext.remap executable directly in this harness. The shift
  // network pass can be added after it understands Rotom-materialized layouts.
  return pm.run(module);
}

tensor_ext::LayoutAttr getArgLayout(func::FuncOp func, unsigned index) {
  auto originalType = func.getArgAttrOfType<tensor_ext::OriginalTypeAttr>(
      index, tensor_ext::TensorExtDialect::kOriginalTypeAttrName);
  if (!originalType) return nullptr;
  return dyn_cast<tensor_ext::LayoutAttr>(originalType.getLayout());
}

tensor_ext::LayoutAttr getResultLayout(func::FuncOp func, unsigned index) {
  auto originalType = func.getResultAttrOfType<tensor_ext::OriginalTypeAttr>(
      index, tensor_ext::TensorExtDialect::kOriginalTypeAttrName);
  if (!originalType) return nullptr;
  return dyn_cast<tensor_ext::LayoutAttr>(originalType.getLayout());
}

std::vector<float> packMatrix(tensor_ext::LayoutAttr layout,
                              RankedTensorType packedType,
                              const std::vector<std::vector<float>>& matrix) {
  std::vector<float> packed(packedType.getNumElements(), 0.0f);
  int64_t numSlots = packedType.getDimSize(1);

  PointPairCollector collector(layout.getIntegerRelation().getNumDomainVars(),
                               /*rangeDims=*/2);
  enumeratePoints(layout.getIntegerRelation(), collector);
  for (const auto& [domain, range] : collector.points) {
    int64_t sourceRow = domain[0];
    int64_t sourceCol = domain[1];
    int64_t targetCiphertext = range[0];
    int64_t targetSlot = range[1];
    packed[targetCiphertext * numSlots + targetSlot] =
        matrix[sourceRow][sourceCol];
  }
  return packed;
}

void expectFloatVectorsNear(const std::vector<float>& actual,
                            const std::vector<float>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], 1e-5) << "at index " << i;
  }
}

TEST(RotomPipelineExecutionTest, RectangularMatmulMatchesReference) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = openfhe::parse(&context, R"mlir(
#layout_lhs = #rotom.layout<dims = [#rotom.dim<[0:2:1]>, #rotom.dim<[1:4:1]>], n = 8>
#layout_rhs = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:2:1]>], n = 8>
#seed_lhs = #rotom.seed<layouts = [#layout_lhs]>
#seed_rhs = #rotom.seed<layouts = [#layout_rhs]>

module {
  func.func @main(%lhs: tensor<2x4xf32> {rotom.seed = #seed_lhs}, %rhs: tensor<4x2xf32> {rotom.seed = #seed_rhs}, %init: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = linalg.matmul ins(%lhs, %rhs : tensor<2x4xf32>, tensor<4x2xf32>) outs(%init : tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runRotomPipeline(module.get(), &context,
                                         /*ciphertextSize=*/8)));

  bool hasMatmul = false;
  module->walk([&](linalg::MatmulOp) { hasMatmul = true; });
  EXPECT_FALSE(hasMatmul);

  func::FuncOp main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  tensor_ext::LayoutAttr lhsLayout = getArgLayout(main, 0);
  tensor_ext::LayoutAttr rhsLayout = getArgLayout(main, 1);
  tensor_ext::LayoutAttr initLayout = getArgLayout(main, 2);
  tensor_ext::LayoutAttr resultLayout = getResultLayout(main, 0);
  ASSERT_TRUE(lhsLayout);
  ASSERT_TRUE(rhsLayout);
  ASSERT_TRUE(initLayout);
  ASSERT_TRUE(resultLayout);

  std::vector<std::vector<float>> lhs = {
      {1, 2, 3, 4},
      {5, 6, 7, 8},
  };
  std::vector<std::vector<float>> rhs = {
      {2, 3},
      {5, 7},
      {11, 13},
      {17, 19},
  };
  std::vector<std::vector<float>> init = {
      {100, 200},
      {300, 400},
  };

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {
      TypedCppValue(packMatrix(
          lhsLayout, cast<RankedTensorType>(main.getArgument(0).getType()),
          lhs)),
      TypedCppValue(packMatrix(
          rhsLayout, cast<RankedTensorType>(main.getArgument(1).getType()),
          rhs)),
      TypedCppValue(packMatrix(
          initLayout, cast<RankedTensorType>(main.getArgument(2).getType()),
          init)),
  };
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);
  ASSERT_EQ(results.size(), 1);

  std::vector<std::vector<float>> expected = {
      {100 + 1 * 2 + 2 * 5 + 3 * 11 + 4 * 17,
       200 + 1 * 3 + 2 * 7 + 3 * 13 + 4 * 19},
      {300 + 5 * 2 + 6 * 5 + 7 * 11 + 8 * 17,
       400 + 5 * 3 + 6 * 7 + 7 * 13 + 8 * 19},
  };
  auto actual = std::get<std::shared_ptr<std::vector<float>>>(results[0].value);
  expectFloatVectorsNear(
      *actual,
      packMatrix(resultLayout,
                 cast<RankedTensorType>(main.getFunctionType().getResult(0)),
                 expected));
}

TEST(RotomPipelineExecutionTest, ElementwiseAddAndMulMatchReference) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = openfhe::parse(&context, R"mlir(
#layout_a = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:4:1]>], n = 16>
#layout_b = #rotom.layout<dims = [#rotom.dim<[1:4:1]>, #rotom.dim<[0:4:1]>], n = 16>
#seed_a = #rotom.seed<layouts = [#layout_a]>
#seed_b = #rotom.seed<layouts = [#layout_b]>

module {
  func.func @main(%a: tensor<4x4xf32> {rotom.seed = #seed_a}, %b: tensor<4x4xf32> {rotom.seed = #seed_b}) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
    %0 = arith.addf %a, %b : tensor<4x4xf32>
    %1 = arith.mulf %a, %b : tensor<4x4xf32>
    return %0, %1 : tensor<4x4xf32>, tensor<4x4xf32>
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runRotomPipeline(module.get(), &context,
                                         /*ciphertextSize=*/16)));

  func::FuncOp main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  tensor_ext::LayoutAttr lhsLayout = getArgLayout(main, 0);
  tensor_ext::LayoutAttr rhsLayout = getArgLayout(main, 1);
  tensor_ext::LayoutAttr addLayout = getResultLayout(main, 0);
  tensor_ext::LayoutAttr mulLayout = getResultLayout(main, 1);
  ASSERT_TRUE(lhsLayout);
  ASSERT_TRUE(rhsLayout);
  ASSERT_TRUE(addLayout);
  ASSERT_TRUE(mulLayout);

  std::vector<std::vector<float>> lhs = {
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
      {13, 14, 15, 16},
  };
  std::vector<std::vector<float>> rhs = {
      {16, 15, 14, 13},
      {12, 11, 10, 9},
      {8, 7, 6, 5},
      {4, 3, 2, 1},
  };
  std::vector<std::vector<float>> expectedAdd(4, std::vector<float>(4));
  std::vector<std::vector<float>> expectedMul(4, std::vector<float>(4));
  for (int64_t i = 0; i < 4; ++i) {
    for (int64_t j = 0; j < 4; ++j) {
      expectedAdd[i][j] = lhs[i][j] + rhs[i][j];
      expectedMul[i][j] = lhs[i][j] * rhs[i][j];
    }
  }

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {
      TypedCppValue(packMatrix(
          lhsLayout, cast<RankedTensorType>(main.getArgument(0).getType()),
          lhs)),
      TypedCppValue(packMatrix(
          rhsLayout, cast<RankedTensorType>(main.getArgument(1).getType()),
          rhs)),
  };
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);
  ASSERT_EQ(results.size(), 2);

  auto actualAdd =
      std::get<std::shared_ptr<std::vector<float>>>(results[0].value);
  auto actualMul =
      std::get<std::shared_ptr<std::vector<float>>>(results[1].value);
  expectFloatVectorsNear(
      *actualAdd,
      packMatrix(addLayout,
                 cast<RankedTensorType>(main.getFunctionType().getResult(0)),
                 expectedAdd));
  expectFloatVectorsNear(
      *actualMul,
      packMatrix(mulLayout,
                 cast<RankedTensorType>(main.getFunctionType().getResult(1)),
                 expectedMul));
}

// Both operands share a rolled (diagonal) layout. Proves a rolled seed flows
// through layout assignment, materialization, and the relation-driven lowering,
// and that packMatrix packs inputs by the rolled relation -- the premise that
// rolled layouts already compute correctly through the existing path.
TEST(RotomPipelineExecutionTest, RolledLayoutElementwiseAddMatchesReference) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = openfhe::parse(&context, R"mlir(
#layout_rolled = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:4:1]>], n = 16, rolls = [(0, 1)]>
#seed_rolled = #rotom.seed<layouts = [#layout_rolled]>

module {
  func.func @main(%a: tensor<4x4xf32> {rotom.seed = #seed_rolled}, %b: tensor<4x4xf32> {rotom.seed = #seed_rolled}) -> tensor<4x4xf32> {
    %0 = arith.addf %a, %b : tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runRotomPipeline(module.get(), &context,
                                         /*ciphertextSize=*/16)));

  func::FuncOp main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  tensor_ext::LayoutAttr lhsLayout = getArgLayout(main, 0);
  tensor_ext::LayoutAttr rhsLayout = getArgLayout(main, 1);
  tensor_ext::LayoutAttr addLayout = getResultLayout(main, 0);
  ASSERT_TRUE(lhsLayout);
  ASSERT_TRUE(rhsLayout);
  ASSERT_TRUE(addLayout);

  std::vector<std::vector<float>> lhs = {
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
      {13, 14, 15, 16},
  };
  std::vector<std::vector<float>> rhs = {
      {16, 15, 14, 13},
      {12, 11, 10, 9},
      {8, 7, 6, 5},
      {4, 3, 2, 1},
  };
  std::vector<std::vector<float>> expectedAdd(4, std::vector<float>(4));
  for (int64_t i = 0; i < 4; ++i) {
    for (int64_t j = 0; j < 4; ++j) {
      expectedAdd[i][j] = lhs[i][j] + rhs[i][j];
    }
  }

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {
      TypedCppValue(packMatrix(
          lhsLayout, cast<RankedTensorType>(main.getArgument(0).getType()),
          lhs)),
      TypedCppValue(packMatrix(
          rhsLayout, cast<RankedTensorType>(main.getArgument(1).getType()),
          rhs)),
  };
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);
  ASSERT_EQ(results.size(), 1);

  auto actualAdd =
      std::get<std::shared_ptr<std::vector<float>>>(results[0].value);
  expectFloatVectorsNear(
      *actualAdd,
      packMatrix(addLayout,
                 cast<RankedTensorType>(main.getFunctionType().getResult(0)),
                 expectedAdd));
}

// One operand is row-major, the other is rolled. The relation-driven lowering
// must remap between the rolled and unrolled packings to align them -- exactly
// the cross-layout alignment a rolled-layout search will depend on.
TEST(RotomPipelineExecutionTest, RolledAndRowMajorAddMatchesReference) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = openfhe::parse(&context, R"mlir(
#layout_row = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:4:1]>], n = 16>
#layout_rolled = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:4:1]>], n = 16, rolls = [(0, 1)]>
#seed_row = #rotom.seed<layouts = [#layout_row]>
#seed_rolled = #rotom.seed<layouts = [#layout_rolled]>

module {
  func.func @main(%a: tensor<4x4xf32> {rotom.seed = #seed_row}, %b: tensor<4x4xf32> {rotom.seed = #seed_rolled}) -> tensor<4x4xf32> {
    %0 = arith.addf %a, %b : tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runRotomPipeline(module.get(), &context,
                                         /*ciphertextSize=*/16)));

  func::FuncOp main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  tensor_ext::LayoutAttr lhsLayout = getArgLayout(main, 0);
  tensor_ext::LayoutAttr rhsLayout = getArgLayout(main, 1);
  tensor_ext::LayoutAttr addLayout = getResultLayout(main, 0);
  ASSERT_TRUE(lhsLayout);
  ASSERT_TRUE(rhsLayout);
  ASSERT_TRUE(addLayout);

  std::vector<std::vector<float>> lhs = {
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
      {13, 14, 15, 16},
  };
  std::vector<std::vector<float>> rhs = {
      {16, 15, 14, 13},
      {12, 11, 10, 9},
      {8, 7, 6, 5},
      {4, 3, 2, 1},
  };
  std::vector<std::vector<float>> expectedAdd(4, std::vector<float>(4));
  for (int64_t i = 0; i < 4; ++i) {
    for (int64_t j = 0; j < 4; ++j) {
      expectedAdd[i][j] = lhs[i][j] + rhs[i][j];
    }
  }

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {
      TypedCppValue(packMatrix(
          lhsLayout, cast<RankedTensorType>(main.getArgument(0).getType()),
          lhs)),
      TypedCppValue(packMatrix(
          rhsLayout, cast<RankedTensorType>(main.getArgument(1).getType()),
          rhs)),
  };
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);
  ASSERT_EQ(results.size(), 1);

  auto actualAdd =
      std::get<std::shared_ptr<std::vector<float>>>(results[0].value);
  expectFloatVectorsNear(
      *actualAdd,
      packMatrix(addLayout,
                 cast<RankedTensorType>(main.getFunctionType().getResult(0)),
                 expectedAdd));
}

int64_t countRotations(ModuleOp module) {
  int64_t count = 0;
  module.walk([&](tensor_ext::RotateOp) { ++count; });
  return count;
}

const char* kRowMajorMatvec = R"mlir(
#a = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:4:1]>], n = 16>
#x = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:1:1]>], n = 16>
#seed_a = #rotom.seed<layouts = [#a]>
#seed_x = #rotom.seed<layouts = [#x]>

module {
  func.func @main(%a: tensor<4x4xf32> {rotom.seed = #seed_a}, %x: tensor<4x1xf32> {rotom.seed = #seed_x}, %init: tensor<4x1xf32>) -> tensor<4x1xf32> {
    %0 = linalg.matmul ins(%a, %x : tensor<4x4xf32>, tensor<4x1xf32>) outs(%init : tensor<4x1xf32>) -> tensor<4x1xf32>
    return %0 : tensor<4x1xf32>
  }
}
)mlir";

const char* kRolledMatvec = R"mlir(
#a = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:4:1]>], n = 16, rolls = [(0, 1)]>
#x = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:1:1]>], n = 16>
#seed_a = #rotom.seed<layouts = [#a]>
#seed_x = #rotom.seed<layouts = [#x]>

module {
  func.func @main(%a: tensor<4x4xf32> {rotom.seed = #seed_a}, %x: tensor<4x1xf32> {rotom.seed = #seed_x}, %init: tensor<4x1xf32>) -> tensor<4x1xf32> {
    %0 = linalg.matmul ins(%a, %x : tensor<4x4xf32>, tensor<4x1xf32>) outs(%init : tensor<4x1xf32>) -> tensor<4x1xf32>
    return %0 : tensor<4x1xf32>
  }
}
)mlir";

// The diagonal (rolled) packing of the matrix is the Halevi-Shoup win: it
// collapses the matvec contraction into far fewer ciphertext rotations than a
// naive row-major lowering. The rotation-aware search now auto-discovers this
// ciphertext-axis diagonal from EITHER seed (row-major or hand-rolled), so both
// lower to the same baby-step/giant-step count -- and far below the O(K^2) a
// brute-force row-major matvec would emit.
TEST(RotomPipelineExecutionTest, RolledMatvecUsesFewerRotationsAndMatchesReference) {
  MLIRContext context;
  initContext(context);

  OwningOpRef<ModuleOp> rowMajor = openfhe::parse(&context, kRowMajorMatvec);
  ASSERT_TRUE(rowMajor);
  ASSERT_TRUE(succeeded(runRotomPipeline(rowMajor.get(), &context,
                                         /*ciphertextSize=*/16)));

  OwningOpRef<ModuleOp> rolled = openfhe::parse(&context, kRolledMatvec);
  ASSERT_TRUE(rolled);
  ASSERT_TRUE(succeeded(runRotomPipeline(rolled.get(), &context,
                                         /*ciphertextSize=*/16)));

  int64_t rowMajorRotations = countRotations(rowMajor.get());
  int64_t rolledRotations = countRotations(rolled.get());
  // Both seeds are auto-diagonalized to the K=4 baby-step/giant-step schedule
  // (4 rotations), well under the brute-force row-major matvec's O(K^2).
  EXPECT_EQ(rowMajorRotations, 4);
  EXPECT_EQ(rolledRotations, 4);

  // The rolled lowering still computes the correct matvec.
  func::FuncOp main = rolled->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  bool hasMatmul = false;
  rolled->walk([&](linalg::MatmulOp) { hasMatmul = true; });
  EXPECT_FALSE(hasMatmul);

  tensor_ext::LayoutAttr aLayout = getArgLayout(main, 0);
  tensor_ext::LayoutAttr xLayout = getArgLayout(main, 1);
  tensor_ext::LayoutAttr initLayout = getArgLayout(main, 2);
  tensor_ext::LayoutAttr resultLayout = getResultLayout(main, 0);
  ASSERT_TRUE(aLayout);
  ASSERT_TRUE(xLayout);
  ASSERT_TRUE(initLayout);
  ASSERT_TRUE(resultLayout);

  std::vector<std::vector<float>> a = {
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
      {13, 14, 15, 16},
  };
  std::vector<std::vector<float>> x = {{1}, {2}, {3}, {4}};
  std::vector<std::vector<float>> init = {{10}, {20}, {30}, {40}};

  Interpreter interpreter(rolled.get());
  std::vector<TypedCppValue> inputs = {
      TypedCppValue(packMatrix(
          aLayout, cast<RankedTensorType>(main.getArgument(0).getType()), a)),
      TypedCppValue(packMatrix(
          xLayout, cast<RankedTensorType>(main.getArgument(1).getType()), x)),
      TypedCppValue(packMatrix(
          initLayout, cast<RankedTensorType>(main.getArgument(2).getType()),
          init)),
  };
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);
  ASSERT_EQ(results.size(), 1);

  std::vector<std::vector<float>> expected = {
      {10 + 1 * 1 + 2 * 2 + 3 * 3 + 4 * 4},
      {20 + 5 * 1 + 6 * 2 + 7 * 3 + 8 * 4},
      {30 + 9 * 1 + 10 * 2 + 11 * 3 + 12 * 4},
      {40 + 13 * 1 + 14 * 2 + 15 * 3 + 16 * 4},
  };
  auto actual = std::get<std::shared_ptr<std::vector<float>>>(results[0].value);
  expectFloatVectorsNear(
      *actual,
      packMatrix(resultLayout,
                 cast<RankedTensorType>(main.getFunctionType().getResult(0)),
                 expected));
}

// The matrix is packed one ciphertext per diagonal (ct = (row - col) mod K,
// slot = col), i.e. the diagonal lives on the CIPHERTEXT axis -- the
// multi-ciphertext Halevi-Shoup packing. RotomTensorOpLowering lowers it with a
// baby-step/giant-step schedule that rotates the shared vector ~sqrt(K) times
// (the BSGS win) rather than rotating both operands per contraction term.
const char* kCtDiagonalMatvec4 = R"mlir(
#a = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:4:1]>], n = 4, rolls = [(0, 1)]>
#x = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:1:1]>], n = 4>
#seed_a = #rotom.seed<layouts = [#a]>
#seed_x = #rotom.seed<layouts = [#x]>

module {
  func.func @main(%a: tensor<4x4xf32> {rotom.seed = #seed_a}, %x: tensor<4x1xf32> {rotom.seed = #seed_x}, %init: tensor<4x1xf32>) -> tensor<4x1xf32> {
    %0 = linalg.matmul ins(%a, %x : tensor<4x4xf32>, tensor<4x1xf32>) outs(%init : tensor<4x1xf32>) -> tensor<4x1xf32>
    return %0 : tensor<4x1xf32>
  }
}
)mlir";

TEST(RotomPipelineExecutionTest, CtDiagonalMatvecBsgsMatchesReference) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = openfhe::parse(&context, kCtDiagonalMatvec4);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runRotomPipeline(module.get(), &context,
                                         /*ciphertextSize=*/4)));

  func::FuncOp main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  bool hasMatmul = false;
  module->walk([&](linalg::MatmulOp) { hasMatmul = true; });
  EXPECT_FALSE(hasMatmul);

  // Baby-step/giant-step schedule for K=4 (baby=giant=2): 1 baby + 1 giant
  // ciphertext-vector rotation (the 2*sqrt(K)-2 = 2 that matter for FHE latency)
  // plus 2 matrix-diagonal rotations that fold to free for a plaintext matrix.
  // The single-ciphertext rolled kernel uses 14 rotations for the same matvec.
  EXPECT_EQ(countRotations(module.get()), 4);

  tensor_ext::LayoutAttr aLayout = getArgLayout(main, 0);
  tensor_ext::LayoutAttr xLayout = getArgLayout(main, 1);
  tensor_ext::LayoutAttr initLayout = getArgLayout(main, 2);
  tensor_ext::LayoutAttr resultLayout = getResultLayout(main, 0);
  ASSERT_TRUE(aLayout);
  ASSERT_TRUE(xLayout);
  ASSERT_TRUE(initLayout);
  ASSERT_TRUE(resultLayout);

  std::vector<std::vector<float>> a = {
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
      {13, 14, 15, 16},
  };
  std::vector<std::vector<float>> x = {{1}, {2}, {3}, {4}};
  std::vector<std::vector<float>> init = {{10}, {20}, {30}, {40}};

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {
      TypedCppValue(packMatrix(
          aLayout, cast<RankedTensorType>(main.getArgument(0).getType()), a)),
      TypedCppValue(packMatrix(
          xLayout, cast<RankedTensorType>(main.getArgument(1).getType()), x)),
      TypedCppValue(packMatrix(
          initLayout, cast<RankedTensorType>(main.getArgument(2).getType()),
          init)),
  };
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);
  ASSERT_EQ(results.size(), 1);

  std::vector<std::vector<float>> expected = {
      {10 + 1 * 1 + 2 * 2 + 3 * 3 + 4 * 4},
      {20 + 5 * 1 + 6 * 2 + 7 * 3 + 8 * 4},
      {30 + 9 * 1 + 10 * 2 + 11 * 3 + 12 * 4},
      {40 + 13 * 1 + 14 * 2 + 15 * 3 + 16 * 4},
  };
  auto actual = std::get<std::shared_ptr<std::vector<float>>>(results[0].value);
  expectFloatVectorsNear(
      *actual,
      packMatrix(resultLayout,
                 cast<RankedTensorType>(main.getFunctionType().getResult(0)),
                 expected));
}

const char* kCtDiagonalMatvec16 = R"mlir(
#a = #rotom.layout<dims = [#rotom.dim<[0:16:1]>, #rotom.dim<[1:16:1]>], n = 16, rolls = [(0, 1)]>
#x = #rotom.layout<dims = [#rotom.dim<[0:16:1]>, #rotom.dim<[1:1:1]>], n = 16>
#seed_a = #rotom.seed<layouts = [#a]>
#seed_x = #rotom.seed<layouts = [#x]>

module {
  func.func @main(%a: tensor<16x16xf32> {rotom.seed = #seed_a}, %x: tensor<16x1xf32> {rotom.seed = #seed_x}, %init: tensor<16x1xf32>) -> tensor<16x1xf32> {
    %0 = linalg.matmul ins(%a, %x : tensor<16x16xf32>, tensor<16x1xf32>) outs(%init : tensor<16x1xf32>) -> tensor<16x1xf32>
    return %0 : tensor<16x1xf32>
  }
}
)mlir";

TEST(RotomPipelineExecutionTest, CtDiagonalMatvecBsgsScalesSublinearly) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = openfhe::parse(&context, kCtDiagonalMatvec16);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runRotomPipeline(module.get(), &context,
                                         /*ciphertextSize=*/16)));

  func::FuncOp main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  bool hasMatmul = false;
  module->walk([&](linalg::MatmulOp) { hasMatmul = true; });
  EXPECT_FALSE(hasMatmul);

  // K=16 (baby=giant=4): 3 baby + 3 giant ciphertext-vector rotations
  // (2*sqrt(16)-2 = 6, the BSGS win) plus 12 matrix-diagonal rotations
  // (foldable for a plaintext matrix), 18 total. A row-major matvec is O(K^2);
  // the diagonal packing keeps the ciphertext-rotation count at O(sqrt(K)).
  EXPECT_EQ(countRotations(module.get()), 18);

  tensor_ext::LayoutAttr aLayout = getArgLayout(main, 0);
  tensor_ext::LayoutAttr xLayout = getArgLayout(main, 1);
  tensor_ext::LayoutAttr initLayout = getArgLayout(main, 2);
  tensor_ext::LayoutAttr resultLayout = getResultLayout(main, 0);
  ASSERT_TRUE(aLayout);
  ASSERT_TRUE(xLayout);
  ASSERT_TRUE(initLayout);
  ASSERT_TRUE(resultLayout);

  std::vector<std::vector<float>> a(16, std::vector<float>(16));
  std::vector<std::vector<float>> x(16, std::vector<float>(1));
  std::vector<std::vector<float>> init(16, std::vector<float>(1));
  for (int64_t i = 0; i < 16; ++i) {
    x[i][0] = static_cast<float>(i + 1);
    init[i][0] = static_cast<float>(i);
    for (int64_t j = 0; j < 16; ++j) {
      a[i][j] = static_cast<float>(i * 16 + j + 1);
    }
  }
  std::vector<std::vector<float>> expected(16, std::vector<float>(1));
  for (int64_t i = 0; i < 16; ++i) {
    float acc = init[i][0];
    for (int64_t j = 0; j < 16; ++j) acc += a[i][j] * x[j][0];
    expected[i][0] = acc;
  }

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {
      TypedCppValue(packMatrix(
          aLayout, cast<RankedTensorType>(main.getArgument(0).getType()), a)),
      TypedCppValue(packMatrix(
          xLayout, cast<RankedTensorType>(main.getArgument(1).getType()), x)),
      TypedCppValue(packMatrix(
          initLayout, cast<RankedTensorType>(main.getArgument(2).getType()),
          init)),
  };
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);
  ASSERT_EQ(results.size(), 1);

  auto actual = std::get<std::shared_ptr<std::vector<float>>>(results[0].value);
  expectFloatVectorsNear(
      *actual,
      packMatrix(resultLayout,
                 cast<RankedTensorType>(main.getFunctionType().getResult(0)),
                 expected));
}

// The matrix is seeded ONLY with a row-major layout (no hand-specified roll),
// exactly what rotom-seed-layout produces. Layout assignment must AUTOMATICALLY
// generate the ciphertext-axis diagonal (rolled) candidate, cost it below
// row-major, select it, and lower via the baby-step/giant-step kernel -- the
// whole point of the search. This is the end-to-end "feed a high-level matvec,
// get BSGS" milestone with no hand-seeding.
const char* kRowMajorSeededMatvec = R"mlir(
#rowA = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:4:1]>], n = 4>
#rowX = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:1:1]>], n = 4>
#seedA = #rotom.seed<layouts = [#rowA]>
#seedX = #rotom.seed<layouts = [#rowX]>

module {
  func.func @main(%a: tensor<4x4xf32> {rotom.seed = #seedA}, %x: tensor<4x1xf32> {rotom.seed = #seedX}, %init: tensor<4x1xf32>) -> tensor<4x1xf32> {
    %0 = linalg.matmul ins(%a, %x : tensor<4x4xf32>, tensor<4x1xf32>) outs(%init : tensor<4x1xf32>) -> tensor<4x1xf32>
    return %0 : tensor<4x1xf32>
  }
}
)mlir";

TEST(RotomPipelineExecutionTest, AutoDiscoversCtDiagonalMatvecFromRowMajorSeed) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module =
      openfhe::parse(&context, kRowMajorSeededMatvec);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runRotomPipeline(module.get(), &context,
                                         /*ciphertextSize=*/4)));

  func::FuncOp main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);

  // The matmul was lowered (the diagonal kernel fired): row-major alone has no
  // kernel for this multi-ciphertext matrix, so a leftover matmul would mean the
  // diagonal was NOT auto-discovered.
  bool hasMatmul = false;
  module->walk([&](linalg::MatmulOp) { hasMatmul = true; });
  EXPECT_FALSE(hasMatmul);

  // Same baby-step/giant-step schedule as the hand-seeded ct-diagonal test:
  // automatic discovery costs no extra rotations.
  EXPECT_EQ(countRotations(module.get()), 4);

  tensor_ext::LayoutAttr aLayout = getArgLayout(main, 0);
  tensor_ext::LayoutAttr xLayout = getArgLayout(main, 1);
  tensor_ext::LayoutAttr initLayout = getArgLayout(main, 2);
  tensor_ext::LayoutAttr resultLayout = getResultLayout(main, 0);
  ASSERT_TRUE(aLayout);
  ASSERT_TRUE(xLayout);
  ASSERT_TRUE(initLayout);
  ASSERT_TRUE(resultLayout);

  std::vector<std::vector<float>> a = {
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
      {13, 14, 15, 16},
  };
  std::vector<std::vector<float>> x = {{1}, {2}, {3}, {4}};
  std::vector<std::vector<float>> init = {{10}, {20}, {30}, {40}};

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {
      TypedCppValue(packMatrix(
          aLayout, cast<RankedTensorType>(main.getArgument(0).getType()), a)),
      TypedCppValue(packMatrix(
          xLayout, cast<RankedTensorType>(main.getArgument(1).getType()), x)),
      TypedCppValue(packMatrix(
          initLayout, cast<RankedTensorType>(main.getArgument(2).getType()),
          init)),
  };
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);
  ASSERT_EQ(results.size(), 1);

  std::vector<std::vector<float>> expected = {
      {10 + 1 * 1 + 2 * 2 + 3 * 3 + 4 * 4},
      {20 + 5 * 1 + 6 * 2 + 7 * 3 + 8 * 4},
      {30 + 9 * 1 + 10 * 2 + 11 * 3 + 12 * 4},
      {40 + 13 * 1 + 14 * 2 + 15 * 3 + 16 * 4},
  };
  auto actual = std::get<std::shared_ptr<std::vector<float>>>(results[0].value);
  expectFloatVectorsNear(
      *actual,
      packMatrix(resultLayout,
                 cast<RankedTensorType>(main.getFunctionType().getResult(0)),
                 expected));
}

// Non-power-of-two matvec A(3x4) * x(4x1) -> out(3x1), seeded only row-major.
// The output rows (3) are not a power of two, so the diagonal layout pads M to 4
// (ct = (row - col) mod 4); the kernel walks the padded 4 diagonals while the
// missing 4th row stays zero (only 3 real rows are packed), so the result is
// still numerically exact. This guards the padded-extent kernel path that real
// networks (e.g. an MNIST 10x... layer) depend on.
const char* kPaddedMatvec3x4 = R"mlir(
#rowA = #rotom.layout<dims = [#rotom.dim<[0:3:1]>, #rotom.dim<[1:4:1]>], n = 4>
#rowX = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:1:1]>], n = 4>
#seedA = #rotom.seed<layouts = [#rowA]>
#seedX = #rotom.seed<layouts = [#rowX]>

module {
  func.func @main(%a: tensor<3x4xf32> {rotom.seed = #seedA}, %x: tensor<4x1xf32> {rotom.seed = #seedX}, %init: tensor<3x1xf32>) -> tensor<3x1xf32> {
    %0 = linalg.matmul ins(%a, %x : tensor<3x4xf32>, tensor<4x1xf32>) outs(%init : tensor<3x1xf32>) -> tensor<3x1xf32>
    return %0 : tensor<3x1xf32>
  }
}
)mlir";

TEST(RotomPipelineExecutionTest, AutoDiscoversPaddedDiagonalMatvec3x4) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = openfhe::parse(&context, kPaddedMatvec3x4);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runRotomPipeline(module.get(), &context,
                                         /*ciphertextSize=*/4)));

  func::FuncOp main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  bool hasMatmul = false;
  module->walk([&](linalg::MatmulOp) { hasMatmul = true; });
  EXPECT_FALSE(hasMatmul);

  tensor_ext::LayoutAttr aLayout = getArgLayout(main, 0);
  tensor_ext::LayoutAttr xLayout = getArgLayout(main, 1);
  tensor_ext::LayoutAttr initLayout = getArgLayout(main, 2);
  tensor_ext::LayoutAttr resultLayout = getResultLayout(main, 0);
  ASSERT_TRUE(aLayout);
  ASSERT_TRUE(xLayout);
  ASSERT_TRUE(initLayout);
  ASSERT_TRUE(resultLayout);

  // The matrix/init/result layouts pad M from 3 to 4, so packMatrix enumerates
  // 4 rows -- provide a zero 4th row (the padding the kernel computes as zero).
  std::vector<std::vector<float>> a = {
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
      {0, 0, 0, 0},
  };
  std::vector<std::vector<float>> x = {{1}, {2}, {3}, {4}};
  std::vector<std::vector<float>> init = {{10}, {20}, {30}, {0}};
  std::vector<std::vector<float>> expected(4, std::vector<float>(1));
  for (int64_t i = 0; i < 4; ++i) {
    float acc = init[i][0];
    for (int64_t k = 0; k < 4; ++k) acc += a[i][k] * x[k][0];
    expected[i][0] = acc;
  }

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {
      TypedCppValue(packMatrix(
          aLayout, cast<RankedTensorType>(main.getArgument(0).getType()), a)),
      TypedCppValue(packMatrix(
          xLayout, cast<RankedTensorType>(main.getArgument(1).getType()), x)),
      TypedCppValue(packMatrix(
          initLayout, cast<RankedTensorType>(main.getArgument(2).getType()),
          init)),
  };
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);
  ASSERT_EQ(results.size(), 1);

  auto actual = std::get<std::shared_ptr<std::vector<float>>>(results[0].value);
  expectFloatVectorsNear(
      *actual,
      packMatrix(resultLayout,
                 cast<RankedTensorType>(main.getFunctionType().getResult(0)),
                 expected));
}

// Non-square (squat) matvec A(2x4) * x(4x1) -> out(2x1), seeded only row-major.
// The search must auto-generate the squat ciphertext-axis diagonal (roll(0,1)
// with unequal extents: ct = (row - col) mod 2), and the squat kernel lowers it
// with a residual rotate-and-sum over the K/M column blocks.
const char* kSquatMatvec2x4 = R"mlir(
#rowA = #rotom.layout<dims = [#rotom.dim<[0:2:1]>, #rotom.dim<[1:4:1]>], n = 4>
#rowX = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:1:1]>], n = 4>
#seedA = #rotom.seed<layouts = [#rowA]>
#seedX = #rotom.seed<layouts = [#rowX]>

module {
  func.func @main(%a: tensor<2x4xf32> {rotom.seed = #seedA}, %x: tensor<4x1xf32> {rotom.seed = #seedX}, %init: tensor<2x1xf32>) -> tensor<2x1xf32> {
    %0 = linalg.matmul ins(%a, %x : tensor<2x4xf32>, tensor<4x1xf32>) outs(%init : tensor<2x1xf32>) -> tensor<2x1xf32>
    return %0 : tensor<2x1xf32>
  }
}
)mlir";

TEST(RotomPipelineExecutionTest, AutoDiscoversSquatDiagonalMatvec2x4) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = openfhe::parse(&context, kSquatMatvec2x4);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runRotomPipeline(module.get(), &context,
                                         /*ciphertextSize=*/4)));

  func::FuncOp main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  bool hasMatmul = false;
  module->walk([&](linalg::MatmulOp) { hasMatmul = true; });
  EXPECT_FALSE(hasMatmul);
  // M=2 diagonals, K/M=2 blocks: per diagonal one residual rotation, plus one
  // diagonal rotation for d=1 (d=0 is a no-op) = 3 rotations.
  EXPECT_EQ(countRotations(module.get()), 3);

  tensor_ext::LayoutAttr aLayout = getArgLayout(main, 0);
  tensor_ext::LayoutAttr xLayout = getArgLayout(main, 1);
  tensor_ext::LayoutAttr initLayout = getArgLayout(main, 2);
  tensor_ext::LayoutAttr resultLayout = getResultLayout(main, 0);
  ASSERT_TRUE(aLayout);
  ASSERT_TRUE(xLayout);
  ASSERT_TRUE(initLayout);
  ASSERT_TRUE(resultLayout);

  std::vector<std::vector<float>> a = {{1, 2, 3, 4}, {5, 6, 7, 8}};
  std::vector<std::vector<float>> x = {{1}, {2}, {3}, {4}};
  std::vector<std::vector<float>> init = {{10}, {20}};

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {
      TypedCppValue(packMatrix(
          aLayout, cast<RankedTensorType>(main.getArgument(0).getType()), a)),
      TypedCppValue(packMatrix(
          xLayout, cast<RankedTensorType>(main.getArgument(1).getType()), x)),
      TypedCppValue(packMatrix(
          initLayout, cast<RankedTensorType>(main.getArgument(2).getType()),
          init)),
  };
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);
  ASSERT_EQ(results.size(), 1);

  std::vector<std::vector<float>> expected = {
      {10 + 1 * 1 + 2 * 2 + 3 * 3 + 4 * 4},
      {20 + 5 * 1 + 6 * 2 + 7 * 3 + 8 * 4},
  };
  auto actual = std::get<std::shared_ptr<std::vector<float>>>(results[0].value);
  expectFloatVectorsNear(
      *actual,
      packMatrix(resultLayout,
                 cast<RankedTensorType>(main.getFunctionType().getResult(0)),
                 expected));
}

const char* kSquatMatvec4x16 = R"mlir(
#rowA = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:16:1]>], n = 16>
#rowX = #rotom.layout<dims = [#rotom.dim<[0:16:1]>, #rotom.dim<[1:1:1]>], n = 16>
#seedA = #rotom.seed<layouts = [#rowA]>
#seedX = #rotom.seed<layouts = [#rowX]>

module {
  func.func @main(%a: tensor<4x16xf32> {rotom.seed = #seedA}, %x: tensor<16x1xf32> {rotom.seed = #seedX}, %init: tensor<4x1xf32>) -> tensor<4x1xf32> {
    %0 = linalg.matmul ins(%a, %x : tensor<4x16xf32>, tensor<16x1xf32>) outs(%init : tensor<4x1xf32>) -> tensor<4x1xf32>
    return %0 : tensor<4x1xf32>
  }
}
)mlir";

TEST(RotomPipelineExecutionTest, AutoDiscoversSquatDiagonalMatvec4x16) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = openfhe::parse(&context, kSquatMatvec4x16);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runRotomPipeline(module.get(), &context,
                                         /*ciphertextSize=*/16)));

  func::FuncOp main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  bool hasMatmul = false;
  module->walk([&](linalg::MatmulOp) { hasMatmul = true; });
  EXPECT_FALSE(hasMatmul);
  // BSGS over D=4 diagonals (baby=giant=2): 1 baby + 1 giant vector rotation + 2
  // matrix-diagonal rotations, plus log2(K/D)=2 residual rotate-and-sum
  // rotations = 6. (The old non-BSGS squat used 11.)
  EXPECT_EQ(countRotations(module.get()), 6);

  tensor_ext::LayoutAttr aLayout = getArgLayout(main, 0);
  tensor_ext::LayoutAttr xLayout = getArgLayout(main, 1);
  tensor_ext::LayoutAttr initLayout = getArgLayout(main, 2);
  tensor_ext::LayoutAttr resultLayout = getResultLayout(main, 0);
  ASSERT_TRUE(aLayout);
  ASSERT_TRUE(xLayout);
  ASSERT_TRUE(initLayout);
  ASSERT_TRUE(resultLayout);

  std::vector<std::vector<float>> a(4, std::vector<float>(16));
  std::vector<std::vector<float>> x(16, std::vector<float>(1));
  std::vector<std::vector<float>> init(4, std::vector<float>(1));
  for (int64_t k = 0; k < 16; ++k) x[k][0] = static_cast<float>(k + 1);
  for (int64_t i = 0; i < 4; ++i) {
    init[i][0] = static_cast<float>(10 * i);
    for (int64_t k = 0; k < 16; ++k) a[i][k] = static_cast<float>(i * 16 + k + 1);
  }
  std::vector<std::vector<float>> expected(4, std::vector<float>(1));
  for (int64_t i = 0; i < 4; ++i) {
    float acc = init[i][0];
    for (int64_t k = 0; k < 16; ++k) acc += a[i][k] * x[k][0];
    expected[i][0] = acc;
  }

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {
      TypedCppValue(packMatrix(
          aLayout, cast<RankedTensorType>(main.getArgument(0).getType()), a)),
      TypedCppValue(packMatrix(
          xLayout, cast<RankedTensorType>(main.getArgument(1).getType()), x)),
      TypedCppValue(packMatrix(
          initLayout, cast<RankedTensorType>(main.getArgument(2).getType()),
          init)),
  };
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);
  ASSERT_EQ(results.size(), 1);

  auto actual = std::get<std::shared_ptr<std::vector<float>>>(results[0].value);
  expectFloatVectorsNear(
      *actual,
      packMatrix(resultLayout,
                 cast<RankedTensorType>(main.getFunctionType().getResult(0)),
                 expected));
}

// Replication: the ciphertext (n = 8) is larger than the contraction (K = 4),
// so the search must replicate the matrix and vector period-K so cyclic
// rotations wrap mod K. Seeded only row-major.
const char* kReplicatedMatvec4x4 = R"mlir(
#rowA = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:4:1]>], n = 8>
#rowX = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:1:1]>], n = 8>
#seedA = #rotom.seed<layouts = [#rowA]>
#seedX = #rotom.seed<layouts = [#rowX]>

module {
  func.func @main(%a: tensor<4x4xf32> {rotom.seed = #seedA}, %x: tensor<4x1xf32> {rotom.seed = #seedX}, %init: tensor<4x1xf32>) -> tensor<4x1xf32> {
    %0 = linalg.matmul ins(%a, %x : tensor<4x4xf32>, tensor<4x1xf32>) outs(%init : tensor<4x1xf32>) -> tensor<4x1xf32>
    return %0 : tensor<4x1xf32>
  }
}
)mlir";

TEST(RotomPipelineExecutionTest, AutoDiscoversReplicatedDiagonalMatvec) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = openfhe::parse(&context, kReplicatedMatvec4x4);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runRotomPipeline(module.get(), &context,
                                         /*ciphertextSize=*/8)));

  func::FuncOp main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  bool hasMatmul = false;
  module->walk([&](linalg::MatmulOp) { hasMatmul = true; });
  EXPECT_FALSE(hasMatmul);
  // Same baby-step/giant-step schedule as the single-period K=4 case (4
  // rotations); replication only adds a gap mask (a multiply, not a rotation).
  EXPECT_EQ(countRotations(module.get()), 4);

  tensor_ext::LayoutAttr aLayout = getArgLayout(main, 0);
  tensor_ext::LayoutAttr xLayout = getArgLayout(main, 1);
  tensor_ext::LayoutAttr initLayout = getArgLayout(main, 2);
  tensor_ext::LayoutAttr resultLayout = getResultLayout(main, 0);
  ASSERT_TRUE(aLayout);
  ASSERT_TRUE(xLayout);
  ASSERT_TRUE(initLayout);
  ASSERT_TRUE(resultLayout);

  std::vector<std::vector<float>> a = {
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
      {13, 14, 15, 16},
  };
  std::vector<std::vector<float>> x = {{1}, {2}, {3}, {4}};
  std::vector<std::vector<float>> init = {{10}, {20}, {30}, {40}};

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {
      TypedCppValue(packMatrix(
          aLayout, cast<RankedTensorType>(main.getArgument(0).getType()), a)),
      TypedCppValue(packMatrix(
          xLayout, cast<RankedTensorType>(main.getArgument(1).getType()), x)),
      TypedCppValue(packMatrix(
          initLayout, cast<RankedTensorType>(main.getArgument(2).getType()),
          init)),
  };
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);
  ASSERT_EQ(results.size(), 1);

  std::vector<std::vector<float>> expected = {
      {10 + 1 * 1 + 2 * 2 + 3 * 3 + 4 * 4},
      {20 + 5 * 1 + 6 * 2 + 7 * 3 + 8 * 4},
      {30 + 9 * 1 + 10 * 2 + 11 * 3 + 12 * 4},
      {40 + 13 * 1 + 14 * 2 + 15 * 3 + 16 * 4},
  };
  auto actual = std::get<std::shared_ptr<std::vector<float>>>(results[0].value);
  expectFloatVectorsNear(
      *actual,
      packMatrix(resultLayout,
                 cast<RankedTensorType>(main.getFunctionType().getResult(0)),
                 expected));
}

// Replication combined with the squat (non-square) case: A(2x4), n = 8.
const char* kReplicatedSquatMatvec2x4 = R"mlir(
#rowA = #rotom.layout<dims = [#rotom.dim<[0:2:1]>, #rotom.dim<[1:4:1]>], n = 8>
#rowX = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:1:1]>], n = 8>
#seedA = #rotom.seed<layouts = [#rowA]>
#seedX = #rotom.seed<layouts = [#rowX]>

module {
  func.func @main(%a: tensor<2x4xf32> {rotom.seed = #seedA}, %x: tensor<4x1xf32> {rotom.seed = #seedX}, %init: tensor<2x1xf32>) -> tensor<2x1xf32> {
    %0 = linalg.matmul ins(%a, %x : tensor<2x4xf32>, tensor<4x1xf32>) outs(%init : tensor<2x1xf32>) -> tensor<2x1xf32>
    return %0 : tensor<2x1xf32>
  }
}
)mlir";

TEST(RotomPipelineExecutionTest, AutoDiscoversReplicatedSquatMatvec) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module =
      openfhe::parse(&context, kReplicatedSquatMatvec2x4);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runRotomPipeline(module.get(), &context,
                                         /*ciphertextSize=*/8)));

  func::FuncOp main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  bool hasMatmul = false;
  module->walk([&](linalg::MatmulOp) { hasMatmul = true; });
  EXPECT_FALSE(hasMatmul);
  // Same as the single-period 2x4 squat: 1 baby + 1 matrix + 1 residual = 3.
  EXPECT_EQ(countRotations(module.get()), 3);

  tensor_ext::LayoutAttr aLayout = getArgLayout(main, 0);
  tensor_ext::LayoutAttr xLayout = getArgLayout(main, 1);
  tensor_ext::LayoutAttr initLayout = getArgLayout(main, 2);
  tensor_ext::LayoutAttr resultLayout = getResultLayout(main, 0);
  ASSERT_TRUE(aLayout);
  ASSERT_TRUE(xLayout);
  ASSERT_TRUE(initLayout);
  ASSERT_TRUE(resultLayout);

  std::vector<std::vector<float>> a = {{1, 2, 3, 4}, {5, 6, 7, 8}};
  std::vector<std::vector<float>> x = {{1}, {2}, {3}, {4}};
  std::vector<std::vector<float>> init = {{10}, {20}};

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {
      TypedCppValue(packMatrix(
          aLayout, cast<RankedTensorType>(main.getArgument(0).getType()), a)),
      TypedCppValue(packMatrix(
          xLayout, cast<RankedTensorType>(main.getArgument(1).getType()), x)),
      TypedCppValue(packMatrix(
          initLayout, cast<RankedTensorType>(main.getArgument(2).getType()),
          init)),
  };
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);
  ASSERT_EQ(results.size(), 1);

  std::vector<std::vector<float>> expected = {
      {10 + 1 * 1 + 2 * 2 + 3 * 3 + 4 * 4},
      {20 + 5 * 1 + 6 * 2 + 7 * 3 + 8 * 4},
  };
  auto actual = std::get<std::shared_ptr<std::vector<float>>>(results[0].value);
  expectFloatVectorsNear(
      *actual,
      packMatrix(resultLayout,
                 cast<RankedTensorType>(main.getFunctionType().getResult(0)),
                 expected));
}

// A larger, MNIST-layer-shaped squat: A(8x64) * x(64) -> out(8) (M << K, like
// the 10x512 / 512x784 MLP layers). This works at scale today -- the matrix is
// D = 8 ciphertexts (one per diagonal) and the contraction K = 64 fits one
// ciphertext -- so no multi-ciphertext "tiling" is required as long as the
// ciphertext holds the contraction (true for MNIST with any realistic degree).
const char* kLargeSquatMatvec8x64 = R"mlir(
#rowA = #rotom.layout<dims = [#rotom.dim<[0:8:1]>, #rotom.dim<[1:64:1]>], n = 64>
#rowX = #rotom.layout<dims = [#rotom.dim<[0:64:1]>, #rotom.dim<[1:1:1]>], n = 64>
#seedA = #rotom.seed<layouts = [#rowA]>
#seedX = #rotom.seed<layouts = [#rowX]>

module {
  func.func @main(%a: tensor<8x64xf32> {rotom.seed = #seedA}, %x: tensor<64x1xf32> {rotom.seed = #seedX}, %init: tensor<8x1xf32>) -> tensor<8x1xf32> {
    %0 = linalg.matmul ins(%a, %x : tensor<8x64xf32>, tensor<64x1xf32>) outs(%init : tensor<8x1xf32>) -> tensor<8x1xf32>
    return %0 : tensor<8x1xf32>
  }
}
)mlir";

TEST(RotomPipelineExecutionTest, AutoDiscoversLargeSquatMatvecAtScale) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = openfhe::parse(&context, kLargeSquatMatvec8x64);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runRotomPipeline(module.get(), &context,
                                         /*ciphertextSize=*/64)));

  func::FuncOp main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  bool hasMatmul = false;
  module->walk([&](linalg::MatmulOp) { hasMatmul = true; });
  EXPECT_FALSE(hasMatmul);
  // Sub-linear in K: a row-major matvec would be O(K^2); this stays O(sqrt(M) +
  // log(K/M)) ciphertext rotations.
  EXPECT_LT(countRotations(module.get()), 64);

  tensor_ext::LayoutAttr aLayout = getArgLayout(main, 0);
  tensor_ext::LayoutAttr xLayout = getArgLayout(main, 1);
  tensor_ext::LayoutAttr initLayout = getArgLayout(main, 2);
  tensor_ext::LayoutAttr resultLayout = getResultLayout(main, 0);
  ASSERT_TRUE(aLayout);
  ASSERT_TRUE(xLayout);
  ASSERT_TRUE(initLayout);
  ASSERT_TRUE(resultLayout);

  std::vector<std::vector<float>> a(8, std::vector<float>(64));
  std::vector<std::vector<float>> x(64, std::vector<float>(1));
  std::vector<std::vector<float>> init(8, std::vector<float>(1));
  for (int64_t k = 0; k < 64; ++k) x[k][0] = static_cast<float>(k + 1);
  for (int64_t i = 0; i < 8; ++i) {
    init[i][0] = static_cast<float>(i);
    for (int64_t k = 0; k < 64; ++k)
      a[i][k] = static_cast<float>((i * 64 + k) % 7 - 3);
  }
  std::vector<std::vector<float>> expected(8, std::vector<float>(1));
  for (int64_t i = 0; i < 8; ++i) {
    float acc = init[i][0];
    for (int64_t k = 0; k < 64; ++k) acc += a[i][k] * x[k][0];
    expected[i][0] = acc;
  }

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {
      TypedCppValue(packMatrix(
          aLayout, cast<RankedTensorType>(main.getArgument(0).getType()), a)),
      TypedCppValue(packMatrix(
          xLayout, cast<RankedTensorType>(main.getArgument(1).getType()), x)),
      TypedCppValue(packMatrix(
          initLayout, cast<RankedTensorType>(main.getArgument(2).getType()),
          init)),
  };
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);
  ASSERT_EQ(results.size(), 1);

  auto actual = std::get<std::shared_ptr<std::vector<float>>>(results[0].value);
  expectFloatVectorsNear(
      *actual,
      packMatrix(resultLayout,
                 cast<RankedTensorType>(main.getFunctionType().getResult(0)),
                 expected));
}

// Dense diagonal packing: at ciphertext size 16 with K=8, P = 16/8 = 2 diagonals
// are packed per ciphertext, so A(4x8) uses 2 ciphertexts (not 4). The search
// auto-discovers it; the dense kernel extracts K-block diagonals and inserts the
// K-wide result into the output. Seeded only row-major.
const char* kDenseMatvec4x8 = R"mlir(
#rowA = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:8:1]>], n = 16>
#rowX = #rotom.layout<dims = [#rotom.dim<[0:8:1]>, #rotom.dim<[1:1:1]>], n = 16>
#seedA = #rotom.seed<layouts = [#rowA]>
#seedX = #rotom.seed<layouts = [#rowX]>

module {
  func.func @main(%a: tensor<4x8xf32> {rotom.seed = #seedA}, %x: tensor<8x1xf32> {rotom.seed = #seedX}, %init: tensor<4x1xf32>) -> tensor<4x1xf32> {
    %0 = linalg.matmul ins(%a, %x : tensor<4x8xf32>, tensor<8x1xf32>) outs(%init : tensor<4x1xf32>) -> tensor<4x1xf32>
    return %0 : tensor<4x1xf32>
  }
}
)mlir";

TEST(RotomPipelineExecutionTest, AutoDiscoversDenseDiagonalMatvec) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = openfhe::parse(&context, kDenseMatvec4x8);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runRotomPipeline(module.get(), &context,
                                         /*ciphertextSize=*/16)));

  func::FuncOp main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  bool hasMatmul = false;
  module->walk([&](linalg::MatmulOp) { hasMatmul = true; });
  EXPECT_FALSE(hasMatmul);
  // The matrix is packed in 2 ciphertexts (P=2 diagonals/ct) -- verified by an
  // extract_slice + insert_slice dense kernel -- using the same baby-step/
  // giant-step rotation count as the single-period case.
  bool hasInsert = false;
  module->walk([&](tensor::InsertSliceOp) { hasInsert = true; });
  EXPECT_TRUE(hasInsert);

  tensor_ext::LayoutAttr aLayout = getArgLayout(main, 0);
  tensor_ext::LayoutAttr xLayout = getArgLayout(main, 1);
  tensor_ext::LayoutAttr initLayout = getArgLayout(main, 2);
  tensor_ext::LayoutAttr resultLayout = getResultLayout(main, 0);
  ASSERT_TRUE(aLayout);
  ASSERT_TRUE(xLayout);
  ASSERT_TRUE(initLayout);
  ASSERT_TRUE(resultLayout);

  std::vector<std::vector<float>> a(4, std::vector<float>(8));
  std::vector<std::vector<float>> x(8, std::vector<float>(1));
  std::vector<std::vector<float>> init(4, std::vector<float>(1));
  for (int64_t k = 0; k < 8; ++k) x[k][0] = static_cast<float>(k + 1);
  for (int64_t i = 0; i < 4; ++i) {
    init[i][0] = static_cast<float>(10 * i);
    for (int64_t k = 0; k < 8; ++k)
      a[i][k] = static_cast<float>((i * 8 + k) % 5 - 2);
  }
  std::vector<std::vector<float>> expected(4, std::vector<float>(1));
  for (int64_t i = 0; i < 4; ++i) {
    float acc = init[i][0];
    for (int64_t k = 0; k < 8; ++k) acc += a[i][k] * x[k][0];
    expected[i][0] = acc;
  }

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {
      TypedCppValue(packMatrix(
          aLayout, cast<RankedTensorType>(main.getArgument(0).getType()), a)),
      TypedCppValue(packMatrix(
          xLayout, cast<RankedTensorType>(main.getArgument(1).getType()), x)),
      TypedCppValue(packMatrix(
          initLayout, cast<RankedTensorType>(main.getArgument(2).getType()),
          init)),
  };
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);
  ASSERT_EQ(results.size(), 1);

  auto actual = std::get<std::shared_ptr<std::vector<float>>>(results[0].value);
  expectFloatVectorsNear(
      *actual,
      packMatrix(resultLayout,
                 cast<RankedTensorType>(main.getFunctionType().getResult(0)),
                 expected));
}

// Dense diagonal packing with a non-power-of-two output dimension: A(6x8) at
// ciphertext size 16 packs P = 16/8 = 2 diagonals per ciphertext while padding
// the 6 output rows up to 8 diagonals. The dense kernel walks the padded 8
// diagonals; the missing 2 rows stay zero (only 6 real rows are packed), so the
// result is exact. This guards the dense kernel's padded-extent path -- the
// regime a real MNIST 10x128 layer hits.
const char* kDensePaddedMatvec6x8 = R"mlir(
#rowA = #rotom.layout<dims = [#rotom.dim<[0:6:1]>, #rotom.dim<[1:8:1]>], n = 16>
#rowX = #rotom.layout<dims = [#rotom.dim<[0:8:1]>, #rotom.dim<[1:1:1]>], n = 16>
#seedA = #rotom.seed<layouts = [#rowA]>
#seedX = #rotom.seed<layouts = [#rowX]>

module {
  func.func @main(%a: tensor<6x8xf32> {rotom.seed = #seedA}, %x: tensor<8x1xf32> {rotom.seed = #seedX}, %init: tensor<6x1xf32>) -> tensor<6x1xf32> {
    %0 = linalg.matmul ins(%a, %x : tensor<6x8xf32>, tensor<8x1xf32>) outs(%init : tensor<6x1xf32>) -> tensor<6x1xf32>
    return %0 : tensor<6x1xf32>
  }
}
)mlir";

TEST(RotomPipelineExecutionTest, AutoDiscoversDensePaddedMatvec6x8) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = openfhe::parse(&context, kDensePaddedMatvec6x8);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runRotomPipeline(module.get(), &context,
                                         /*ciphertextSize=*/16)));

  func::FuncOp main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  bool hasMatmul = false;
  module->walk([&](linalg::MatmulOp) { hasMatmul = true; });
  EXPECT_FALSE(hasMatmul);
  bool hasInsert = false;
  module->walk([&](tensor::InsertSliceOp) { hasInsert = true; });
  EXPECT_TRUE(hasInsert);

  tensor_ext::LayoutAttr aLayout = getArgLayout(main, 0);
  tensor_ext::LayoutAttr xLayout = getArgLayout(main, 1);
  tensor_ext::LayoutAttr initLayout = getArgLayout(main, 2);
  tensor_ext::LayoutAttr resultLayout = getResultLayout(main, 0);
  ASSERT_TRUE(aLayout);
  ASSERT_TRUE(xLayout);
  ASSERT_TRUE(initLayout);
  ASSERT_TRUE(resultLayout);

  // The matrix/init/result layouts pad M from 6 to 8 -- packMatrix enumerates 8
  // rows, so provide two zero padding rows (the kernel computes them as zero).
  std::vector<std::vector<float>> a(8, std::vector<float>(8, 0.0f));
  std::vector<std::vector<float>> x(8, std::vector<float>(1));
  std::vector<std::vector<float>> init(8, std::vector<float>(1, 0.0f));
  for (int64_t k = 0; k < 8; ++k) x[k][0] = static_cast<float>(k + 1);
  for (int64_t i = 0; i < 6; ++i) {
    init[i][0] = static_cast<float>(5 * i);
    for (int64_t k = 0; k < 8; ++k)
      a[i][k] = static_cast<float>((i * 8 + k) % 5 - 2);
  }
  std::vector<std::vector<float>> expected(8, std::vector<float>(1));
  for (int64_t i = 0; i < 8; ++i) {
    float acc = init[i][0];
    for (int64_t k = 0; k < 8; ++k) acc += a[i][k] * x[k][0];
    expected[i][0] = acc;
  }

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {
      TypedCppValue(packMatrix(
          aLayout, cast<RankedTensorType>(main.getArgument(0).getType()), a)),
      TypedCppValue(packMatrix(
          xLayout, cast<RankedTensorType>(main.getArgument(1).getType()), x)),
      TypedCppValue(packMatrix(
          initLayout, cast<RankedTensorType>(main.getArgument(2).getType()),
          init)),
  };
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);
  ASSERT_EQ(results.size(), 1);

  auto actual = std::get<std::shared_ptr<std::vector<float>>>(results[0].value);
  expectFloatVectorsNear(
      *actual,
      packMatrix(resultLayout,
                 cast<RankedTensorType>(main.getFunctionType().getResult(0)),
                 expected));
}

// End-to-end two-layer MLP: h = relu_sq(W1 x + b1); y = W2 h + b2, with a
// square activation (the FHE-friendly CryptoNets nonlinearity). Both matvecs
// are seeded only row-major -- the search must auto-discover the ciphertext-axis
// diagonal kernel for BOTH layers (layer 1 is a 4x4 square matvec, layer 2 a
// 2x4 squat matvec), the square activation runs elementwise on layer 1's packed
// output, and that output must feed layer 2's vector input without an explicit
// re-layout. This is the smallest network that exercises matvec -> activation ->
// matvec layout chaining end-to-end through the Rotom pipeline.
const char* kTwoLayerMlp = R"mlir(
#rowW1 = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:4:1]>], n = 4>
#rowX  = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:1:1]>], n = 4>
#rowW2 = #rotom.layout<dims = [#rotom.dim<[0:2:1]>, #rotom.dim<[1:4:1]>], n = 4>
#seedW1 = #rotom.seed<layouts = [#rowW1]>
#seedX  = #rotom.seed<layouts = [#rowX]>
#seedW2 = #rotom.seed<layouts = [#rowW2]>

module {
  func.func @main(%w1: tensor<4x4xf32> {rotom.seed = #seedW1},
                  %x: tensor<4x1xf32> {rotom.seed = #seedX},
                  %b1: tensor<4x1xf32>,
                  %w2: tensor<2x4xf32> {rotom.seed = #seedW2},
                  %b2: tensor<2x1xf32>) -> tensor<2x1xf32> {
    %h = linalg.matmul ins(%w1, %x : tensor<4x4xf32>, tensor<4x1xf32>) outs(%b1 : tensor<4x1xf32>) -> tensor<4x1xf32>
    %a = arith.mulf %h, %h : tensor<4x1xf32>
    %y = linalg.matmul ins(%w2, %a : tensor<2x4xf32>, tensor<4x1xf32>) outs(%b2 : tensor<2x1xf32>) -> tensor<2x1xf32>
    return %y : tensor<2x1xf32>
  }
}
)mlir";

TEST(RotomPipelineExecutionTest, TwoLayerMlpMatchesReference) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = openfhe::parse(&context, kTwoLayerMlp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runRotomPipeline(module.get(), &context,
                                         /*ciphertextSize=*/4)));

  func::FuncOp main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);

  // Both matmuls must have been lowered (the diagonal kernel fired for each):
  // a leftover matmul would mean a layer was not auto-discovered.
  bool hasMatmul = false;
  module->walk([&](linalg::MatmulOp) { hasMatmul = true; });
  EXPECT_FALSE(hasMatmul);

  // Layer 1 (4x4 square) costs 4 baby-step/giant-step rotations; layer 2 (2x4
  // squat) costs 3; the elementwise square adds none -> 7 ciphertext rotations
  // for the whole network.
  EXPECT_EQ(countRotations(module.get()), 7);

  tensor_ext::LayoutAttr w1Layout = getArgLayout(main, 0);
  tensor_ext::LayoutAttr xLayout = getArgLayout(main, 1);
  tensor_ext::LayoutAttr b1Layout = getArgLayout(main, 2);
  tensor_ext::LayoutAttr w2Layout = getArgLayout(main, 3);
  tensor_ext::LayoutAttr b2Layout = getArgLayout(main, 4);
  tensor_ext::LayoutAttr resultLayout = getResultLayout(main, 0);
  ASSERT_TRUE(w1Layout);
  ASSERT_TRUE(xLayout);
  ASSERT_TRUE(b1Layout);
  ASSERT_TRUE(w2Layout);
  ASSERT_TRUE(b2Layout);
  ASSERT_TRUE(resultLayout);

  std::vector<std::vector<float>> w1 = {
      {0.5f, -0.25f, 1.0f, 0.0f},
      {0.0f, 0.5f, -0.5f, 0.25f},
      {1.0f, 0.0f, 0.0f, -1.0f},
      {-0.5f, 0.25f, 0.5f, 0.5f},
  };
  std::vector<std::vector<float>> x = {{1.0f}, {2.0f}, {-1.0f}, {0.5f}};
  std::vector<std::vector<float>> b1 = {{0.1f}, {-0.2f}, {0.3f}, {0.0f}};
  std::vector<std::vector<float>> w2 = {
      {1.0f, 0.5f, -0.5f, 0.25f},
      {-0.25f, 1.0f, 0.0f, 0.5f},
  };
  std::vector<std::vector<float>> b2 = {{0.2f}, {-0.1f}};

  // h = W1 x + b1; a = h .* h; y = W2 a + b2.
  std::vector<float> h(4), a(4);
  for (int64_t i = 0; i < 4; ++i) {
    float acc = b1[i][0];
    for (int64_t k = 0; k < 4; ++k) acc += w1[i][k] * x[k][0];
    h[i] = acc;
    a[i] = acc * acc;
  }
  std::vector<std::vector<float>> expected(2, std::vector<float>(1));
  for (int64_t j = 0; j < 2; ++j) {
    float acc = b2[j][0];
    for (int64_t i = 0; i < 4; ++i) acc += w2[j][i] * a[i];
    expected[j][0] = acc;
  }

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {
      TypedCppValue(packMatrix(
          w1Layout, cast<RankedTensorType>(main.getArgument(0).getType()), w1)),
      TypedCppValue(packMatrix(
          xLayout, cast<RankedTensorType>(main.getArgument(1).getType()), x)),
      TypedCppValue(packMatrix(
          b1Layout, cast<RankedTensorType>(main.getArgument(2).getType()), b1)),
      TypedCppValue(packMatrix(
          w2Layout, cast<RankedTensorType>(main.getArgument(3).getType()), w2)),
      TypedCppValue(packMatrix(
          b2Layout, cast<RankedTensorType>(main.getArgument(4).getType()), b2)),
  };
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);
  ASSERT_EQ(results.size(), 1);

  auto actual = std::get<std::shared_ptr<std::vector<float>>>(results[0].value);
  expectFloatVectorsNear(
      *actual,
      packMatrix(resultLayout,
                 cast<RankedTensorType>(main.getFunctionType().getResult(0)),
                 expected));
}

// MNIST-shaped two-layer MLP at scale, packed exactly like the reference Rotom
// MNIST (dense ciphertext-axis diagonals): h = W1 x + b1 (64x128 matvec, dense
// at ciphertext size 256 -> P = 256/128 = 2 diagonals/ct), a = h .* h (square
// activation), y = W2 a + b2 (16x64 matvec, dense P = 256/64 = 4). Both layers
// auto-discover the dense diagonal kernel and chain through the square. This is
// the same packing scheme the reference MNIST uses (its 512x784 layer-1 weight
// packs as 16 ciphertexts of 32 diagonals each at ciphertext size 32768); the
// dimensions here are scaled so the cleartext interpreter can unroll every
// diagonal op (full MNIST's 512 diagonals are tractable only on a compiled FHE
// backend, not this op-by-op interpreter).
const char* kDenseMnistScaleMlp = R"mlir(
#rowW1 = #rotom.layout<dims = [#rotom.dim<[0:64:1]>, #rotom.dim<[1:128:1]>], n = 256>
#rowX  = #rotom.layout<dims = [#rotom.dim<[0:128:1]>, #rotom.dim<[1:1:1]>], n = 256>
#rowW2 = #rotom.layout<dims = [#rotom.dim<[0:16:1]>, #rotom.dim<[1:64:1]>], n = 256>
#seedW1 = #rotom.seed<layouts = [#rowW1]>
#seedX  = #rotom.seed<layouts = [#rowX]>
#seedW2 = #rotom.seed<layouts = [#rowW2]>

module {
  func.func @main(%w1: tensor<64x128xf32> {rotom.seed = #seedW1}, %x: tensor<128x1xf32> {rotom.seed = #seedX}, %b1: tensor<64x1xf32>, %w2: tensor<16x64xf32> {rotom.seed = #seedW2}, %b2: tensor<16x1xf32>) -> tensor<16x1xf32> {
    %h = linalg.matmul ins(%w1, %x : tensor<64x128xf32>, tensor<128x1xf32>) outs(%b1 : tensor<64x1xf32>) -> tensor<64x1xf32>
    %a = arith.mulf %h, %h : tensor<64x1xf32>
    %y = linalg.matmul ins(%w2, %a : tensor<16x64xf32>, tensor<64x1xf32>) outs(%b2 : tensor<16x1xf32>) -> tensor<16x1xf32>
    return %y : tensor<16x1xf32>
  }
}
)mlir";

TEST(RotomPipelineExecutionTest, AutoDiscoversDenseMnistScaleMlp) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = openfhe::parse(&context, kDenseMnistScaleMlp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runRotomPipeline(module.get(), &context,
                                         /*ciphertextSize=*/256)));

  func::FuncOp main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  // Both matvecs lowered (no leftover matmul) and both used the dense kernel
  // (insert_slice present -- the K-wide dense result placement).
  bool hasMatmul = false;
  module->walk([&](linalg::MatmulOp) { hasMatmul = true; });
  EXPECT_FALSE(hasMatmul);
  bool hasInsert = false;
  module->walk([&](tensor::InsertSliceOp) { hasInsert = true; });
  EXPECT_TRUE(hasInsert);

  tensor_ext::LayoutAttr w1Layout = getArgLayout(main, 0);
  tensor_ext::LayoutAttr xLayout = getArgLayout(main, 1);
  tensor_ext::LayoutAttr b1Layout = getArgLayout(main, 2);
  tensor_ext::LayoutAttr w2Layout = getArgLayout(main, 3);
  tensor_ext::LayoutAttr b2Layout = getArgLayout(main, 4);
  tensor_ext::LayoutAttr resultLayout = getResultLayout(main, 0);
  ASSERT_TRUE(w1Layout);
  ASSERT_TRUE(xLayout);
  ASSERT_TRUE(b1Layout);
  ASSERT_TRUE(w2Layout);
  ASSERT_TRUE(b2Layout);
  ASSERT_TRUE(resultLayout);

  // Small deterministic weights keep h (and h^2) order-one so float rounding
  // stays well within tolerance across the ~190 accumulations.
  std::vector<std::vector<float>> w1(64, std::vector<float>(128));
  std::vector<std::vector<float>> x(128, std::vector<float>(1));
  std::vector<std::vector<float>> b1(64, std::vector<float>(1));
  std::vector<std::vector<float>> w2(16, std::vector<float>(64));
  std::vector<std::vector<float>> b2(16, std::vector<float>(1));
  for (int64_t i = 0; i < 64; ++i) {
    b1[i][0] = static_cast<float>((i % 3) - 1) * 0.1f;
    for (int64_t k = 0; k < 128; ++k)
      w1[i][k] = static_cast<float>(((i * 128 + k) % 7) - 3) * 0.02f;
  }
  for (int64_t k = 0; k < 128; ++k)
    x[k][0] = static_cast<float>((k % 5) - 2) * 0.2f;
  for (int64_t j = 0; j < 16; ++j) {
    b2[j][0] = static_cast<float>((j % 3) - 1) * 0.1f;
    for (int64_t i = 0; i < 64; ++i)
      w2[j][i] = static_cast<float>(((j * 64 + i) % 5) - 2) * 0.03f;
  }

  // h = W1 x + b1; a = h .* h; y = W2 a + b2.
  std::vector<float> h(64), a(64);
  for (int64_t i = 0; i < 64; ++i) {
    float acc = b1[i][0];
    for (int64_t k = 0; k < 128; ++k) acc += w1[i][k] * x[k][0];
    h[i] = acc;
    a[i] = acc * acc;
  }
  std::vector<std::vector<float>> expected(16, std::vector<float>(1));
  for (int64_t j = 0; j < 16; ++j) {
    float acc = b2[j][0];
    for (int64_t i = 0; i < 64; ++i) acc += w2[j][i] * a[i];
    expected[j][0] = acc;
  }

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {
      TypedCppValue(packMatrix(
          w1Layout, cast<RankedTensorType>(main.getArgument(0).getType()), w1)),
      TypedCppValue(packMatrix(
          xLayout, cast<RankedTensorType>(main.getArgument(1).getType()), x)),
      TypedCppValue(packMatrix(
          b1Layout, cast<RankedTensorType>(main.getArgument(2).getType()), b1)),
      TypedCppValue(packMatrix(
          w2Layout, cast<RankedTensorType>(main.getArgument(3).getType()), w2)),
      TypedCppValue(packMatrix(
          b2Layout, cast<RankedTensorType>(main.getArgument(4).getType()), b2)),
  };
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);
  ASSERT_EQ(results.size(), 1);

  auto actual = std::get<std::shared_ptr<std::vector<float>>>(results[0].value);
  expectFloatVectorsNear(
      *actual,
      packMatrix(resultLayout,
                 cast<RankedTensorType>(main.getFunctionType().getResult(0)),
                 expected));
}

// Full MNIST-scale matvec dimensions (512 hidden, 1024 input -- the padded
// MNIST layer-1 shape) through the dense kernel: h = W1 x + b1 (512x1024, dense
// P=8), a = h .* h, y = W2 a + b2 (16x512, dense P=16). This is the real MNIST
// layer-1 packing (512x784 pads K to 1024; at ciphertext size 8192 the matrix
// is 64 ciphertexts of 8 diagonals). Tractable only because the convert-pass
// layout verification is O(one enumeration per layout) -- per-element ISL
// enumeration made this run for minutes.
const char* kDenseMnistLayerMlp = R"mlir(
#rowW1 = #rotom.layout<dims = [#rotom.dim<[0:512:1]>, #rotom.dim<[1:1024:1]>], n = 8192>
#rowX  = #rotom.layout<dims = [#rotom.dim<[0:1024:1]>, #rotom.dim<[1:1:1]>], n = 8192>
#rowW2 = #rotom.layout<dims = [#rotom.dim<[0:16:1]>, #rotom.dim<[1:512:1]>], n = 8192>
#seedW1 = #rotom.seed<layouts = [#rowW1]>
#seedX  = #rotom.seed<layouts = [#rowX]>
#seedW2 = #rotom.seed<layouts = [#rowW2]>

module {
  func.func @main(%w1: tensor<512x1024xf32> {rotom.seed = #seedW1}, %x: tensor<1024x1xf32> {rotom.seed = #seedX}, %b1: tensor<512x1xf32>, %w2: tensor<16x512xf32> {rotom.seed = #seedW2}, %b2: tensor<16x1xf32>) -> tensor<16x1xf32> {
    %h = linalg.matmul ins(%w1, %x : tensor<512x1024xf32>, tensor<1024x1xf32>) outs(%b1 : tensor<512x1xf32>) -> tensor<512x1xf32>
    %a = arith.mulf %h, %h : tensor<512x1xf32>
    %y = linalg.matmul ins(%w2, %a : tensor<16x512xf32>, tensor<512x1xf32>) outs(%b2 : tensor<16x1xf32>) -> tensor<16x1xf32>
    return %y : tensor<16x1xf32>
  }
}
)mlir";

TEST(RotomPipelineExecutionTest, AutoDiscoversDenseMnistLayerMlp) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = openfhe::parse(&context, kDenseMnistLayerMlp);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runRotomPipeline(module.get(), &context,
                                         /*ciphertextSize=*/8192)));

  func::FuncOp main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  bool hasMatmul = false;
  module->walk([&](linalg::MatmulOp) { hasMatmul = true; });
  EXPECT_FALSE(hasMatmul);

  tensor_ext::LayoutAttr w1Layout = getArgLayout(main, 0);
  tensor_ext::LayoutAttr xLayout = getArgLayout(main, 1);
  tensor_ext::LayoutAttr b1Layout = getArgLayout(main, 2);
  tensor_ext::LayoutAttr w2Layout = getArgLayout(main, 3);
  tensor_ext::LayoutAttr b2Layout = getArgLayout(main, 4);
  tensor_ext::LayoutAttr resultLayout = getResultLayout(main, 0);
  ASSERT_TRUE(w1Layout && xLayout && b1Layout && w2Layout && b2Layout &&
              resultLayout);

  std::vector<std::vector<float>> w1(512, std::vector<float>(1024));
  std::vector<std::vector<float>> x(1024, std::vector<float>(1));
  std::vector<std::vector<float>> b1(512, std::vector<float>(1));
  std::vector<std::vector<float>> w2(16, std::vector<float>(512));
  std::vector<std::vector<float>> b2(16, std::vector<float>(1));
  for (int64_t i = 0; i < 512; ++i) {
    b1[i][0] = static_cast<float>((i % 3) - 1) * 0.05f;
    for (int64_t k = 0; k < 1024; ++k)
      w1[i][k] = static_cast<float>(((i * 1024 + k) % 7) - 3) * 0.004f;
  }
  for (int64_t k = 0; k < 1024; ++k)
    x[k][0] = static_cast<float>((k % 5) - 2) * 0.1f;
  for (int64_t j = 0; j < 16; ++j) {
    b2[j][0] = static_cast<float>((j % 3) - 1) * 0.05f;
    for (int64_t i = 0; i < 512; ++i)
      w2[j][i] = static_cast<float>(((j * 512 + i) % 5) - 2) * 0.01f;
  }

  std::vector<float> h(512), a(512);
  for (int64_t i = 0; i < 512; ++i) {
    float acc = b1[i][0];
    for (int64_t k = 0; k < 1024; ++k) acc += w1[i][k] * x[k][0];
    h[i] = acc;
    a[i] = acc * acc;
  }
  std::vector<std::vector<float>> expected(16, std::vector<float>(1));
  for (int64_t j = 0; j < 16; ++j) {
    float acc = b2[j][0];
    for (int64_t i = 0; i < 512; ++i) acc += w2[j][i] * a[i];
    expected[j][0] = acc;
  }

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {
      TypedCppValue(packMatrix(
          w1Layout, cast<RankedTensorType>(main.getArgument(0).getType()), w1)),
      TypedCppValue(packMatrix(
          xLayout, cast<RankedTensorType>(main.getArgument(1).getType()), x)),
      TypedCppValue(packMatrix(
          b1Layout, cast<RankedTensorType>(main.getArgument(2).getType()), b1)),
      TypedCppValue(packMatrix(
          w2Layout, cast<RankedTensorType>(main.getArgument(3).getType()), w2)),
      TypedCppValue(packMatrix(
          b2Layout, cast<RankedTensorType>(main.getArgument(4).getType()), b2)),
  };
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);
  ASSERT_EQ(results.size(), 1);

  auto actual = std::get<std::shared_ptr<std::vector<float>>>(results[0].value);
  expectFloatVectorsNear(
      *actual,
      packMatrix(resultLayout,
                 cast<RankedTensorType>(main.getFunctionType().getResult(0)),
                 expected));
}

// Under-filled diagonal: ciphertext size 64 with K=8 and only M=4 diagonals, so
// the dense factor P = 64/8 = 8 exceeds the 4 diagonals -- the packed block
// (4*8 = 32 slots) does not fill the 64-slot ciphertext. The matrix layout
// occupies slots [0, 32) with the rest a gap (zero), and the kernel still
// extracts and reduces the 4 diagonals correctly. This is the regime the seed
// pass's replication-to-fill enables at scale (e.g. the MNIST 10x512 layer at
// ciphertext size 32768, where 16 diagonals * 512 = 8192 < 32768).
const char* kUnderfilledMatvec4x8 = R"mlir(
#diagA = #rotom.layout<n = 64, rolls = [(0, 1)], dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:8:1]>]>
#idX = #rotom.layout<n = 64, dims = [#rotom.dim<[0:8:1]>, #rotom.dim<[1:1:1]>]>
#seedA = #rotom.seed<layouts = [#diagA]>
#seedX = #rotom.seed<layouts = [#idX]>

module {
  func.func @main(%a: tensor<4x8xf32> {rotom.seed = #seedA}, %x: tensor<8x1xf32> {rotom.seed = #seedX}, %init: tensor<4x1xf32>) -> tensor<4x1xf32> {
    %0 = linalg.matmul ins(%a, %x : tensor<4x8xf32>, tensor<8x1xf32>) outs(%init : tensor<4x1xf32>) -> tensor<4x1xf32>
    return %0 : tensor<4x1xf32>
  }
}
)mlir";

TEST(RotomPipelineExecutionTest, UnderfilledDiagonalMatvecMatchesReference) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = openfhe::parse(&context, kUnderfilledMatvec4x8);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runRotomPipeline(module.get(), &context,
                                         /*ciphertextSize=*/64)));

  func::FuncOp main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  bool hasMatmul = false;
  module->walk([&](linalg::MatmulOp) { hasMatmul = true; });
  EXPECT_FALSE(hasMatmul);

  tensor_ext::LayoutAttr aLayout = getArgLayout(main, 0);
  tensor_ext::LayoutAttr xLayout = getArgLayout(main, 1);
  tensor_ext::LayoutAttr initLayout = getArgLayout(main, 2);
  tensor_ext::LayoutAttr resultLayout = getResultLayout(main, 0);
  ASSERT_TRUE(aLayout && xLayout && initLayout && resultLayout);

  std::vector<std::vector<float>> a = {
      {1, 2, 3, 4, 5, 6, 7, 8},
      {8, 7, 6, 5, 4, 3, 2, 1},
      {2, 0, -1, 1, 3, -2, 0, 1},
      {-1, 1, -1, 1, -1, 1, -1, 1},
  };
  std::vector<std::vector<float>> x = {{1}, {-1}, {2}, {0}, {1}, {1}, {-2}, {3}};
  std::vector<std::vector<float>> init = {{1}, {2}, {3}, {4}};
  std::vector<std::vector<float>> expected(4, std::vector<float>(1));
  for (int64_t i = 0; i < 4; ++i) {
    float acc = init[i][0];
    for (int64_t k = 0; k < 8; ++k) acc += a[i][k] * x[k][0];
    expected[i][0] = acc;
  }

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {
      TypedCppValue(packMatrix(
          aLayout, cast<RankedTensorType>(main.getArgument(0).getType()), a)),
      TypedCppValue(packMatrix(
          xLayout, cast<RankedTensorType>(main.getArgument(1).getType()), x)),
      TypedCppValue(packMatrix(
          initLayout, cast<RankedTensorType>(main.getArgument(2).getType()),
          init)),
  };
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);
  ASSERT_EQ(results.size(), 1);

  auto actual = std::get<std::shared_ptr<std::vector<float>>>(results[0].value);
  expectFloatVectorsNear(
      *actual,
      packMatrix(resultLayout,
                 cast<RankedTensorType>(main.getFunctionType().getResult(0)),
                 expected));
}

// Dense diagonal with a non-power-of-two contraction K: a 2x6 matvec at
// ciphertext size 16. K=6 pads to Kp=8 (each diagonal is an 8-wide block, the
// last 2 columns zero), and P = 16/8 = 2 diagonals pack per ciphertext. This
// guards the dense kernel's padded-K path -- the regime the MNIST 512x784 layer
// hits at ciphertext size 32768 (K=784 -> 1024).
const char* kDensePaddedKMatvec2x6 = R"mlir(
#diagA = #rotom.layout<n = 16, rolls = [(0, 1)], dims = [#rotom.dim<[0:2:1]>, #rotom.dim<[1:6:1]>]>
#idX = #rotom.layout<n = 16, dims = [#rotom.dim<[0:6:1]>, #rotom.dim<[1:1:1]>]>
#seedA = #rotom.seed<layouts = [#diagA]>
#seedX = #rotom.seed<layouts = [#idX]>

module {
  func.func @main(%a: tensor<2x6xf32> {rotom.seed = #seedA}, %x: tensor<6x1xf32> {rotom.seed = #seedX}, %init: tensor<2x1xf32>) -> tensor<2x1xf32> {
    %0 = linalg.matmul ins(%a, %x : tensor<2x6xf32>, tensor<6x1xf32>) outs(%init : tensor<2x1xf32>) -> tensor<2x1xf32>
    return %0 : tensor<2x1xf32>
  }
}
)mlir";

TEST(RotomPipelineExecutionTest, DensePaddedKMatvecMatchesReference) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module =
      openfhe::parse(&context, kDensePaddedKMatvec2x6);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runRotomPipeline(module.get(), &context,
                                         /*ciphertextSize=*/16)));

  func::FuncOp main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  bool hasMatmul = false;
  module->walk([&](linalg::MatmulOp) { hasMatmul = true; });
  EXPECT_FALSE(hasMatmul);
  bool hasInsert = false;
  module->walk([&](tensor::InsertSliceOp) { hasInsert = true; });
  EXPECT_TRUE(hasInsert);

  tensor_ext::LayoutAttr aLayout = getArgLayout(main, 0);
  tensor_ext::LayoutAttr xLayout = getArgLayout(main, 1);
  tensor_ext::LayoutAttr initLayout = getArgLayout(main, 2);
  tensor_ext::LayoutAttr resultLayout = getResultLayout(main, 0);
  ASSERT_TRUE(aLayout && xLayout && initLayout && resultLayout);

  // K=6 pads to Kp=8: packMatrix enumerates 8 columns, so provide two zero
  // padding columns (and matching zero vector entries) -- the kernel computes
  // them as zero.
  std::vector<std::vector<float>> a = {
      {1, 2, 3, 4, 5, 6, 0, 0},
      {-1, 1, -2, 2, -3, 3, 0, 0},
  };
  std::vector<std::vector<float>> x = {{2}, {-1}, {1}, {3}, {-2}, {1}, {0}, {0}};
  std::vector<std::vector<float>> init = {{5}, {-5}};
  std::vector<std::vector<float>> expected(2, std::vector<float>(1));
  for (int64_t i = 0; i < 2; ++i) {
    float acc = init[i][0];
    for (int64_t k = 0; k < 8; ++k) acc += a[i][k] * x[k][0];
    expected[i][0] = acc;
  }

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {
      TypedCppValue(packMatrix(
          aLayout, cast<RankedTensorType>(main.getArgument(0).getType()), a)),
      TypedCppValue(packMatrix(
          xLayout, cast<RankedTensorType>(main.getArgument(1).getType()), x)),
      TypedCppValue(packMatrix(
          initLayout, cast<RankedTensorType>(main.getArgument(2).getType()),
          init)),
  };
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);
  ASSERT_EQ(results.size(), 1);

  auto actual = std::get<std::shared_ptr<std::vector<float>>>(results[0].value);
  expectFloatVectorsNear(
      *actual,
      packMatrix(resultLayout,
                 cast<RankedTensorType>(main.getFunctionType().getResult(0)),
                 expected));
}

}  // namespace
}  // namespace heir
}  // namespace mlir
