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
// collapses the matvec contraction into far fewer ciphertext rotations than the
// row-major packing, while computing the same result through the rotation
// kernel. This is the payoff that lets a rolled-layout search pay off.
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
  EXPECT_GT(rowMajorRotations, 0);
  EXPECT_LT(rolledRotations, rowMajorRotations)
      << "rolled diagonal packing should use fewer rotations: rolled="
      << rolledRotations << " row-major=" << rowMajorRotations;

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

}  // namespace
}  // namespace heir
}  // namespace mlir
