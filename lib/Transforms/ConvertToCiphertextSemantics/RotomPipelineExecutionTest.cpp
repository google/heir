#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Dialect/Rotom/IR/RotomDialect.h"
#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/LayoutAssignment.h"
#include "lib/Dialect/Rotom/Transforms/MaterializeTensorExtLayout/MaterializeTensorExtLayout.h"
#include "lib/Dialect/Rotom/Transforms/OutlineKernels/OutlineKernels.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Dialect/TensorExt/Transforms/ImplementShiftNetwork.h"
#include "lib/Target/OpenFhePke/Interpreter.h"
#include "lib/Transforms/ConvertToCiphertextSemantics/ConvertToCiphertextSemantics.h"
#include "lib/Utils/Layout/Utils.h"
#include "mlir/include/mlir/Dialect/Func/Extensions/InlinerExtension.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OwningOpRef.h"            // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"         // from @llvm-project

namespace mlir {
namespace heir {
namespace {

using openfhe::Interpreter;
using openfhe::TypedCppValue;

void initContext(MLIRContext& context) {
  openfhe::initContext(context);
  context.loadDialect<rotom::RotomDialect, secret::SecretDialect>();
  // The pipeline inlines outlined Rotom kernel functions before
  // interpretation (the interpreter has no call support).
  DialectRegistry registry;
  func::registerInlinerExtension(registry);
  context.appendDialectRegistry(registry);
}

LogicalResult runRotomPipeline(ModuleOp module, MLIRContext* context,
                               int ciphertextSize) {
  PassManager pm(context);
  pm.addPass(rotom::createLayoutAssignment());
  pm.addPass(rotom::createOutlineKernels());
  pm.addPass(rotom::createMaterializeTensorExtLayout());

  ConvertToCiphertextSemanticsOptions options;
  options.ciphertextSize = ciphertextSize;
  pm.addPass(createConvertToCiphertextSemantics(options));
  // Inline the outlined kernel functions (the interpreter has no call
  // support), then lower the tensor_ext.convert_layout ops the Rotom
  // lowerings emit into rotations + masks.
  pm.addPass(createInlinerPass());
  pm.addPass(tensor_ext::createImplementShiftNetwork());
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
    // A vector layout has a single domain var; read it as (row, 0).
    int64_t sourceRow = domain[0];
    int64_t sourceCol = domain.size() > 1 ? domain[1] : 0;
    int64_t targetCiphertext = range[0];
    int64_t targetSlot = range[1];
    packed[targetCiphertext * numSlots + targetSlot] =
        matrix[sourceRow][sourceCol];
  }
  return packed;
}

// Compares only the positions the layout claims. A result layout with gap
// pieces (e.g. the matmul result, whose summed-k offsets hold unspecified
// window sums) makes no statement about the other slots.
void expectClaimedSlotsMatch(const std::vector<float>& actual,
                             tensor_ext::LayoutAttr layout,
                             RankedTensorType packedType,
                             const std::vector<std::vector<float>>& expected) {
  int64_t numSlots = packedType.getDimSize(1);
  PointPairCollector collector(layout.getIntegerRelation().getNumDomainVars(),
                               /*rangeDims=*/2);
  enumeratePoints(layout.getIntegerRelation(), collector);
  ASSERT_FALSE(collector.points.empty());
  for (const auto& [domain, range] : collector.points) {
    int64_t sourceRow = domain[0];
    int64_t sourceCol = domain.size() > 1 ? domain[1] : 0;
    int64_t index = range[0] * numSlots + range[1];
    ASSERT_LT(index, static_cast<int64_t>(actual.size()));
    EXPECT_NEAR(actual[index], expected[sourceRow][sourceCol], 1e-4)
        << "at (" << sourceRow << ", " << sourceCol << ") -> packed index "
        << index;
  }
}

void expectFloatVectorsNear(const std::vector<float>& actual,
                            const std::vector<float>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], 1e-5) << "at index " << i;
  }
}

TEST(RotomPipelineExecutionTest, ElementwiseAddSubMulMatchReference) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = openfhe::parse(&context, R"mlir(
#layout_a = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:4:1]>], n = 16>
#layout_b = #rotom.layout<dims = [#rotom.dim<[1:4:1]>, #rotom.dim<[0:4:1]>], n = 16>
#seed_a = #rotom.seed<layouts = [#layout_a]>
#seed_b = #rotom.seed<layouts = [#layout_b]>

module {
  func.func @main(%a: tensor<4x4xf32> {rotom.seed = #seed_a}, %b: tensor<4x4xf32> {rotom.seed = #seed_b}) -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) {
    %0 = arith.addf %a, %b : tensor<4x4xf32>
    %1 = arith.mulf %a, %b : tensor<4x4xf32>
    %2 = arith.subf %a, %b : tensor<4x4xf32>
    return %0, %1, %2 : tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>
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
  tensor_ext::LayoutAttr subLayout = getResultLayout(main, 2);
  ASSERT_TRUE(lhsLayout);
  ASSERT_TRUE(rhsLayout);
  ASSERT_TRUE(addLayout);
  ASSERT_TRUE(mulLayout);
  ASSERT_TRUE(subLayout);

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
  std::vector<std::vector<float>> expectedSub(4, std::vector<float>(4));
  for (int64_t i = 0; i < 4; ++i) {
    for (int64_t j = 0; j < 4; ++j) {
      expectedAdd[i][j] = lhs[i][j] + rhs[i][j];
      expectedMul[i][j] = lhs[i][j] * rhs[i][j];
      expectedSub[i][j] = lhs[i][j] - rhs[i][j];
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
  ASSERT_EQ(results.size(), 3);

  auto actualAdd =
      std::get<std::shared_ptr<std::vector<float>>>(results[0].value);
  auto actualMul =
      std::get<std::shared_ptr<std::vector<float>>>(results[1].value);
  auto actualSub =
      std::get<std::shared_ptr<std::vector<float>>>(results[2].value);
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
  expectFloatVectorsNear(
      *actualSub,
      packMatrix(subLayout,
                 cast<RankedTensorType>(main.getFunctionType().getResult(2)),
                 expectedSub));
}

// A diamond: one shared value (%s = a + b) feeds two consumers. Exercises the
// DAG path of layout assignment -- the shared value is assigned a single layout
// (its cone is counted once in the accumulated cost) and both consumers convert
// onto their compute layout -- and proves the result is numerically correct.
TEST(RotomPipelineExecutionTest, SharedValueDiamondMatchesReference) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = openfhe::parse(&context, R"mlir(
#layout_a = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:4:1]>], n = 16>
#layout_b = #rotom.layout<dims = [#rotom.dim<[1:4:1]>, #rotom.dim<[0:4:1]>], n = 16>
#seed_a = #rotom.seed<layouts = [#layout_a]>
#seed_b = #rotom.seed<layouts = [#layout_b]>

module {
  func.func @main(%a: tensor<4x4xf32> {rotom.seed = #seed_a}, %b: tensor<4x4xf32> {rotom.seed = #seed_b},
                  %p: tensor<4x4xf32> {rotom.seed = #seed_a}, %q: tensor<4x4xf32> {rotom.seed = #seed_b})
      -> (tensor<4x4xf32>, tensor<4x4xf32>) {
    %s = arith.addf %a, %b : tensor<4x4xf32>
    %l = arith.addf %s, %p : tensor<4x4xf32>
    %r = arith.mulf %s, %q : tensor<4x4xf32>
    return %l, %r : tensor<4x4xf32>, tensor<4x4xf32>
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runRotomPipeline(module.get(), &context,
                                         /*ciphertextSize=*/16)));

  func::FuncOp main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  tensor_ext::LayoutAttr aLayout = getArgLayout(main, 0);
  tensor_ext::LayoutAttr bLayout = getArgLayout(main, 1);
  tensor_ext::LayoutAttr pLayout = getArgLayout(main, 2);
  tensor_ext::LayoutAttr qLayout = getArgLayout(main, 3);
  tensor_ext::LayoutAttr lLayout = getResultLayout(main, 0);
  tensor_ext::LayoutAttr rLayout = getResultLayout(main, 1);
  ASSERT_TRUE(aLayout && bLayout && pLayout && qLayout && lLayout && rLayout);

  std::vector<std::vector<float>> a = {
      {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
  std::vector<std::vector<float>> b = {
      {16, 15, 14, 13}, {12, 11, 10, 9}, {8, 7, 6, 5}, {4, 3, 2, 1}};
  std::vector<std::vector<float>> p = {
      {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}};
  std::vector<std::vector<float>> q = {
      {2, 0, 2, 0}, {0, 2, 0, 2}, {2, 0, 2, 0}, {0, 2, 0, 2}};
  std::vector<std::vector<float>> expectedL(4, std::vector<float>(4));
  std::vector<std::vector<float>> expectedR(4, std::vector<float>(4));
  for (int64_t i = 0; i < 4; ++i) {
    for (int64_t j = 0; j < 4; ++j) {
      float s = a[i][j] + b[i][j];
      expectedL[i][j] = s + p[i][j];
      expectedR[i][j] = s * q[i][j];
    }
  }

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {
      TypedCppValue(packMatrix(
          aLayout, cast<RankedTensorType>(main.getArgument(0).getType()), a)),
      TypedCppValue(packMatrix(
          bLayout, cast<RankedTensorType>(main.getArgument(1).getType()), b)),
      TypedCppValue(packMatrix(
          pLayout, cast<RankedTensorType>(main.getArgument(2).getType()), p)),
      TypedCppValue(packMatrix(
          qLayout, cast<RankedTensorType>(main.getArgument(3).getType()), q)),
  };
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);
  ASSERT_EQ(results.size(), 2);

  auto actualL =
      std::get<std::shared_ptr<std::vector<float>>>(results[0].value);
  auto actualR =
      std::get<std::shared_ptr<std::vector<float>>>(results[1].value);
  expectFloatVectorsNear(
      *actualL,
      packMatrix(lLayout,
                 cast<RankedTensorType>(main.getFunctionType().getResult(0)),
                 expectedL));
  expectFloatVectorsNear(
      *actualR,
      packMatrix(rLayout,
                 cast<RankedTensorType>(main.getFunctionType().getResult(1)),
                 expectedR));
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

// Roll-free matvec through the full pipeline: align (replicate the vector
// across the matrix rows' slot positions), multiply once, rotate-and-reduce
// k. The result layout claims only the k=0 offsets, which must hold the true
// row sums.
TEST(RotomPipelineExecutionTest, MatvecMatchesReference) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = openfhe::parse(&context, R"mlir(
#seed_mat = #rotom.seed<layouts = [#rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:4:1]>], n = 16>]>
#seed_col = #rotom.seed<layouts = [#rotom.layout<dims = [#rotom.dim<[0:4:1]>], n = 16>]>

module {
  func.func @main(%a: tensor<4x4xf32> {rotom.seed = #seed_mat}, %b: tensor<4x1xf32> {rotom.seed = #seed_col}) -> tensor<4x1xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<4x1xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x1xf32>) -> tensor<4x1xf32>
    %0 = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x1xf32>) outs(%fill : tensor<4x1xf32>) -> tensor<4x1xf32>
    return %0 : tensor<4x1xf32>
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runRotomPipeline(module.get(), &context,
                                         /*ciphertextSize=*/16)));

  func::FuncOp main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  tensor_ext::LayoutAttr matLayout = getArgLayout(main, 0);
  tensor_ext::LayoutAttr vecLayout = getArgLayout(main, 1);
  tensor_ext::LayoutAttr resultLayout = getResultLayout(main, 0);
  ASSERT_TRUE(matLayout);
  ASSERT_TRUE(vecLayout);
  ASSERT_TRUE(resultLayout);

  std::vector<std::vector<float>> mat = {
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
      {13, 14, 15, 16},
  };
  std::vector<std::vector<float>> vec = {{2}, {1}, {3}, {5}};
  std::vector<std::vector<float>> expected(4, std::vector<float>(1, 0.0f));
  for (int64_t i = 0; i < 4; ++i) {
    for (int64_t k = 0; k < 4; ++k) {
      expected[i][0] += mat[i][k] * vec[k][0];
    }
  }

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {
      TypedCppValue(packMatrix(
          matLayout, cast<RankedTensorType>(main.getArgument(0).getType()),
          mat)),
      TypedCppValue(packMatrix(
          vecLayout, cast<RankedTensorType>(main.getArgument(1).getType()),
          vec)),
  };
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);
  ASSERT_EQ(results.size(), 1);

  auto actual = std::get<std::shared_ptr<std::vector<float>>>(results[0].value);
  expectClaimedSlotsMatch(
      *actual, resultLayout,
      cast<RankedTensorType>(main.getFunctionType().getResult(0)), expected);
}

// Roll-free 4x4 matmul at n=64 (the whole iteration space in one
// ciphertext's slots): both operands convert onto their expanded placements,
// one multiply, rotate-and-reduce k. Claimed result slots must equal the
// plaintext matmul.
TEST(RotomPipelineExecutionTest, MatmulMatchesReference) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = openfhe::parse(&context, R"mlir(
#seed_row = #rotom.seed<layouts = [#rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:4:1]>], n = 64>]>

module {
  func.func @main(%a: tensor<4x4xf32> {rotom.seed = #seed_row}, %b: tensor<4x4xf32> {rotom.seed = #seed_row}) -> tensor<4x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<4x4xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x4xf32>) -> tensor<4x4xf32>
    %0 = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%fill : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runRotomPipeline(module.get(), &context,
                                         /*ciphertextSize=*/64)));

  func::FuncOp main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  tensor_ext::LayoutAttr lhsLayout = getArgLayout(main, 0);
  tensor_ext::LayoutAttr rhsLayout = getArgLayout(main, 1);
  tensor_ext::LayoutAttr resultLayout = getResultLayout(main, 0);
  ASSERT_TRUE(lhsLayout);
  ASSERT_TRUE(rhsLayout);
  ASSERT_TRUE(resultLayout);

  std::vector<std::vector<float>> lhs = {
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
      {13, 14, 15, 16},
  };
  std::vector<std::vector<float>> rhs = {
      {2, 0, 1, 3},
      {1, 4, 0, 2},
      {0, 1, 2, 1},
      {3, 2, 1, 0},
  };
  std::vector<std::vector<float>> expected(4, std::vector<float>(4, 0.0f));
  for (int64_t i = 0; i < 4; ++i) {
    for (int64_t j = 0; j < 4; ++j) {
      for (int64_t k = 0; k < 4; ++k) {
        expected[i][j] += lhs[i][k] * rhs[k][j];
      }
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

  auto actual = std::get<std::shared_ptr<std::vector<float>>>(results[0].value);
  expectClaimedSlotsMatch(
      *actual, resultLayout,
      cast<RankedTensorType>(main.getFunctionType().getResult(0)), expected);
}

// The same 4x4 matmul at n=16: the whole 64-point iteration space needs 4
// ciphertexts from single-ciphertext operands, so the operand expansion
// changes the ciphertext count -- the explicit rotate/mask/accumulate path
// that convert_layout cannot express. Priced by the search, emitted by the
// lowering, and numerically exact.
TEST(RotomPipelineExecutionTest, MatmulAtSmallCiphertextMatchesReference) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = openfhe::parse(&context, R"mlir(
#seed_row = #rotom.seed<layouts = [#rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:4:1]>], n = 16>]>

module {
  func.func @main(%a: tensor<4x4xf32> {rotom.seed = #seed_row}, %b: tensor<4x4xf32> {rotom.seed = #seed_row}) -> tensor<4x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<4x4xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x4xf32>) -> tensor<4x4xf32>
    %0 = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%fill : tensor<4x4xf32>) -> tensor<4x4xf32>
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
  tensor_ext::LayoutAttr resultLayout = getResultLayout(main, 0);
  ASSERT_TRUE(lhsLayout);
  ASSERT_TRUE(rhsLayout);
  ASSERT_TRUE(resultLayout);

  std::vector<std::vector<float>> lhs = {
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
      {13, 14, 15, 16},
  };
  std::vector<std::vector<float>> rhs = {
      {2, 0, 1, 3},
      {1, 4, 0, 2},
      {0, 1, 2, 1},
      {3, 2, 1, 0},
  };
  std::vector<std::vector<float>> expected(4, std::vector<float>(4, 0.0f));
  for (int64_t i = 0; i < 4; ++i) {
    for (int64_t j = 0; j < 4; ++j) {
      for (int64_t k = 0; k < 4; ++k) {
        expected[i][j] += lhs[i][k] * rhs[k][j];
      }
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

  auto actual = std::get<std::shared_ptr<std::vector<float>>>(results[0].value);
  expectClaimedSlotsMatch(
      *actual, resultLayout,
      cast<RankedTensorType>(main.getFunctionType().getResult(0)), expected);
}

}  // namespace
}  // namespace heir
}  // namespace mlir
