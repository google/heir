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

// The operands are seeded at DIFFERENT ciphertext counts: one row-per-
// ciphertext (4 cts via an explicit slot gap), one row-major (1 ct). Whatever
// compute layout the search picks, one operand crosses a ciphertext-count
// boundary -- the conversion tensor_ext.convert_layout cannot express -- so
// this exercises the explicit rotate/mask/accumulate conversion (expansion or
// compaction) inside an elementwise kernel.
TEST(RotomPipelineExecutionTest,
     ElementwiseAcrossCiphertextCountsMatchesReference) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = openfhe::parse(&context, R"mlir(
#layout_ct_rows = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[G:4:1]>, #rotom.dim<[1:4:1]>], n = 16>
#layout_row = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:4:1]>], n = 16>
#seed_ct_rows = #rotom.seed<layouts = [#layout_ct_rows]>
#seed_row = #rotom.seed<layouts = [#layout_row]>

module {
  func.func @main(%a: tensor<4x4xf32> {rotom.seed = #seed_ct_rows}, %b: tensor<4x4xf32> {rotom.seed = #seed_row}) -> tensor<4x4xf32> {
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
  expectClaimedSlotsMatch(
      *actualAdd, addLayout,
      cast<RankedTensorType>(main.getFunctionType().getResult(0)), expectedAdd);
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
// ciphertexts, and both operands are seeded sources, so the search repacks
// them at a ciphertext-k compute placement directly (4 ciphertexts each,
// no ciphertext-space conversions) and the k-sum is plain ciphertext adds.
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

  // The sources were repacked away from their single-ciphertext row-major
  // seeds onto the 4-ciphertext compute placement.
  EXPECT_EQ(cast<RankedTensorType>(main.getArgument(0).getType()).getDimSize(0),
            4);
  EXPECT_EQ(cast<RankedTensorType>(main.getArgument(1).getType()).getDimSize(0),
            4);

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

// A rectangular 4x8 by 8x4 matmul: non-square k flows through plan
// enumeration (roll extents need not match), assignment, and lowering.
TEST(RotomPipelineExecutionTest, MatmulRectangularMatchesReference) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = openfhe::parse(&context, R"mlir(
#seed_a = #rotom.seed<layouts = [#rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:8:1]>], n = 32>]>
#seed_b = #rotom.seed<layouts = [#rotom.layout<dims = [#rotom.dim<[0:8:1]>, #rotom.dim<[1:4:1]>], n = 32>]>

module {
  func.func @main(%a: tensor<4x8xf32> {rotom.seed = #seed_a}, %b: tensor<8x4xf32> {rotom.seed = #seed_b}) -> tensor<4x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<4x4xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x4xf32>) -> tensor<4x4xf32>
    %0 = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x4xf32>) outs(%fill : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runRotomPipeline(module.get(), &context,
                                         /*ciphertextSize=*/32)));

  func::FuncOp main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  tensor_ext::LayoutAttr lhsLayout = getArgLayout(main, 0);
  tensor_ext::LayoutAttr rhsLayout = getArgLayout(main, 1);
  tensor_ext::LayoutAttr resultLayout = getResultLayout(main, 0);
  ASSERT_TRUE(lhsLayout);
  ASSERT_TRUE(rhsLayout);
  ASSERT_TRUE(resultLayout);

  std::vector<std::vector<float>> lhs(4, std::vector<float>(8));
  std::vector<std::vector<float>> rhs(8, std::vector<float>(4));
  for (int64_t i = 0; i < 4; ++i) {
    for (int64_t k = 0; k < 8; ++k) lhs[i][k] = 1 + ((3 * i + k) % 7);
  }
  for (int64_t k = 0; k < 8; ++k) {
    for (int64_t j = 0; j < 4; ++j) rhs[k][j] = 1 + ((2 * k + 5 * j) % 5);
  }
  std::vector<std::vector<float>> expected(4, std::vector<float>(4, 0.0f));
  for (int64_t i = 0; i < 4; ++i) {
    for (int64_t j = 0; j < 4; ++j) {
      for (int64_t k = 0; k < 8; ++k) {
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

// Rectangular elementwise: a 2x4 row-major and column-major seed pair. Each
// source adopts the diagonal single-roll variant of its own seed -- an
// UNEQUAL-extent roll (the 4-extent piece rolled by the 2-extent one) --
// and the rolled layouts pack, align, and unpack correctly end to end.
TEST(RotomPipelineExecutionTest, ElementwiseUnequalExtentDiagonalAdoption) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = openfhe::parse(&context, R"mlir(
#seed_a = #rotom.seed<layouts = [#rotom.layout<dims = [#rotom.dim<[0:2:1]>, #rotom.dim<[1:4:1]>], n = 8>]>
#seed_b = #rotom.seed<layouts = [#rotom.layout<dims = [#rotom.dim<[1:4:1]>, #rotom.dim<[0:2:1]>], n = 8>]>

module {
  func.func @main(%a: tensor<2x4xf32> {rotom.seed = #seed_a}, %b: tensor<2x4xf32> {rotom.seed = #seed_b}) -> tensor<2x4xf32> {
    %0 = arith.addf %a, %b : tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(runRotomPipeline(module.get(), &context,
                                         /*ciphertextSize=*/8)));

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
  };
  std::vector<std::vector<float>> rhs = {
      {10, 20, 30, 40},
      {50, 60, 70, 80},
  };
  std::vector<std::vector<float>> expected(2, std::vector<float>(4));
  for (int64_t i = 0; i < 2; ++i) {
    for (int64_t j = 0; j < 4; ++j) {
      expected[i][j] = lhs[i][j] + rhs[i][j];
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

// A chain (A x B) x C: the intermediate cannot repack, so the second matmul
// prices hosting it. The search picks a rolled ct-diagonal plan for the
// second matmul that hosts the first's result at its assigned layout (zero
// conversion for the chained value beyond the replicate-then-roll fill) and
// repacks C at the matching rolled expansion -- rolled hosting keeps a
// matmul chain cheap end to end.
TEST(RotomPipelineExecutionTest, MatmulChainHostsIntermediateMatchesReference) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = openfhe::parse(&context, R"mlir(
#seed_row = #rotom.seed<layouts = [#rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:4:1]>], n = 16>]>

module {
  func.func @main(%a: tensor<4x4xf32> {rotom.seed = #seed_row}, %b: tensor<4x4xf32> {rotom.seed = #seed_row}, %c: tensor<4x4xf32> {rotom.seed = #seed_row}) -> tensor<4x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<4x4xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x4xf32>) -> tensor<4x4xf32>
    %0 = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%fill : tensor<4x4xf32>) -> tensor<4x4xf32>
    %empty2 = tensor.empty() : tensor<4x4xf32>
    %fill2 = linalg.fill ins(%cst : f32) outs(%empty2 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %1 = linalg.matmul ins(%0, %c : tensor<4x4xf32>, tensor<4x4xf32>) outs(%fill2 : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
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
  tensor_ext::LayoutAttr cLayout = getArgLayout(main, 2);
  tensor_ext::LayoutAttr resultLayout = getResultLayout(main, 0);
  ASSERT_TRUE(aLayout);
  ASSERT_TRUE(bLayout);
  ASSERT_TRUE(cLayout);
  ASSERT_TRUE(resultLayout);

  std::vector<std::vector<float>> a = {
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
      {13, 14, 15, 16},
  };
  std::vector<std::vector<float>> b = {
      {2, 0, 1, 3},
      {1, 4, 0, 2},
      {0, 1, 2, 1},
      {3, 2, 1, 0},
  };
  std::vector<std::vector<float>> c = {
      {1, 0, 2, 0},
      {0, 3, 0, 1},
      {2, 0, 1, 0},
      {0, 1, 0, 2},
  };
  std::vector<std::vector<float>> ab(4, std::vector<float>(4, 0.0f));
  std::vector<std::vector<float>> expected(4, std::vector<float>(4, 0.0f));
  for (int64_t i = 0; i < 4; ++i) {
    for (int64_t j = 0; j < 4; ++j) {
      for (int64_t k = 0; k < 4; ++k) {
        ab[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  for (int64_t i = 0; i < 4; ++i) {
    for (int64_t j = 0; j < 4; ++j) {
      for (int64_t k = 0; k < 4; ++k) {
        expected[i][j] += ab[i][k] * c[k][j];
      }
    }
  }

  Interpreter interpreter(module.get());
  std::vector<TypedCppValue> inputs = {
      TypedCppValue(packMatrix(
          aLayout, cast<RankedTensorType>(main.getArgument(0).getType()), a)),
      TypedCppValue(packMatrix(
          bLayout, cast<RankedTensorType>(main.getArgument(1).getType()), b)),
      TypedCppValue(packMatrix(
          cLayout, cast<RankedTensorType>(main.getArgument(2).getType()), c)),
  };
  std::vector<TypedCppValue> results = interpreter.interpret("main", inputs);
  ASSERT_EQ(results.size(), 1);

  auto actual = std::get<std::shared_ptr<std::vector<float>>>(results[0].value);
  expectClaimedSlotsMatch(
      *actual, resultLayout,
      cast<RankedTensorType>(main.getFunctionType().getResult(0)), expected);
}

// A matmul whose lhs is an intermediate (a + a), not a source: intermediates
// cannot repack at encode time, so the lowering must still convert the lhs
// onto its expanded compute placement in ciphertext space -- the priced
// conversion path -- while the rhs source repacks for free.
TEST(RotomPipelineExecutionTest, MatmulOfIntermediateConvertsOperand) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = openfhe::parse(&context, R"mlir(
#seed_row = #rotom.seed<layouts = [#rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:4:1]>], n = 16>]>

module {
  func.func @main(%a: tensor<4x4xf32> {rotom.seed = #seed_row}, %b: tensor<4x4xf32> {rotom.seed = #seed_row}) -> tensor<4x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %s = arith.addf %a, %a : tensor<4x4xf32>
    %empty = tensor.empty() : tensor<4x4xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x4xf32>) -> tensor<4x4xf32>
    %0 = linalg.matmul ins(%s, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%fill : tensor<4x4xf32>) -> tensor<4x4xf32>
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
        expected[i][j] += 2 * lhs[i][k] * rhs[k][j];
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

// The rolled ct-diagonal matmul: operands seeded at the aligned pair
//   a = roll(0,2) [1:4:1];[0:4:1][R:4:1]  (ciphertext c holds A[i, (c+r)%4])
//   b = roll(0,2) [0:4:1];[R:4:1][1:4:1]  (ciphertext c holds B[(c+j)%4, j])
// so ciphertext c of the product holds A[x,(c+y)%4] * B[(c+y)%4,y] at slot
// (x, y) and the k-sum is three plain ciphertext adds -- no rotations, no
// masks -- leaving the full result row-major in one ciphertext. Both
// expansions equal the seeds, so the zero-conversion rolled plan wins the
// search outright.
TEST(RotomPipelineExecutionTest, MatmulRolledCtDiagonalMatchesReference) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = openfhe::parse(&context, R"mlir(
#layout_a = #rotom.layout<dims = [#rotom.dim<[1:4:1]>, #rotom.dim<[0:4:1]>, #rotom.dim<[R:4:1]>], n = 16, rolls = [(0, 2)]>
#layout_b = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[R:4:1]>, #rotom.dim<[1:4:1]>], n = 16, rolls = [(0, 2)]>
#seed_a = #rotom.seed<layouts = [#layout_a]>
#seed_b = #rotom.seed<layouts = [#layout_b]>

module {
  func.func @main(%a: tensor<4x4xf32> {rotom.seed = #seed_a}, %b: tensor<4x4xf32> {rotom.seed = #seed_b}) -> tensor<4x4xf32> {
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

// The compact-source route: the lhs is seeded column-major only (one
// ciphertext) and the rhs at its rolled ct-diagonal placement. The
// ciphertext-count carrying cost keeps the lhs COMPACT at its seed -- the
// winning plan expands it in ciphertext space by replicate-then-roll
// (ciphertext replication is free; the roll-by-replication placement with
// the partner outermost in slots makes each expanded ciphertext one whole
// -ciphertext rotation of the source, no masks) rather than repacking the
// source across four ciphertexts at encode time.
TEST(RotomPipelineExecutionTest, MatmulCompactSourceReplicatesThenRolls) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = openfhe::parse(&context, R"mlir(
#layout_col = #rotom.layout<dims = [#rotom.dim<[1:4:1]>, #rotom.dim<[0:4:1]>], n = 16>
#layout_b = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:4:1]>, #rotom.dim<[R:4:1]>], n = 16, rolls = [(0, 1)]>
#seed_a = #rotom.seed<layouts = [#layout_col]>
#seed_b = #rotom.seed<layouts = [#layout_b]>

module {
  func.func @main(%a: tensor<4x4xf32> {rotom.seed = #seed_a}, %b: tensor<4x4xf32> {rotom.seed = #seed_b}) -> tensor<4x4xf32> {
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

  // The compact lhs stays at one ciphertext; the pre-rolled rhs stays at its
  // four-ciphertext seed.
  EXPECT_EQ(cast<RankedTensorType>(main.getArgument(0).getType()).getDimSize(0),
            1);
  EXPECT_EQ(cast<RankedTensorType>(main.getArgument(1).getType()).getDimSize(0),
            4);

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

// The slot-diagonal (Halevi-Shoup) matmul. The matmul's lhs is an
// INTERMEDIATE (a + a) held at the one-ciphertext diagonal packing
// roll(0,1) [1:4];[0:4] -- slot (k', i) holds A[i, (k'+i)%4], diagonal k'
// at offset i -- so it cannot repack at encode time; its only cheap
// expansion is onto the slot-diagonal plan's lhs placement
// roll(1,2) [R:4];[1:4][0:4], the same diagonal content replicated across
// the four j ciphertexts (four zero-shift full-row copies, free). The rhs
// is seeded at the plan's expanded placement. The kernel multiplies once
// and sums k by the slot rotate-and-reduce over the rolled piece; the
// result keeps the [j:ct];[G][i] shape, four ciphertexts with the true
// sums at the k'=0 offsets.
TEST(RotomPipelineExecutionTest, MatmulSlotDiagonalHaleviShoupStyle) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = openfhe::parse(&context, R"mlir(
#layout_diag = #rotom.layout<dims = [#rotom.dim<[1:4:1]>, #rotom.dim<[0:4:1]>], n = 16, rolls = [(0, 1)]>
#layout_b = #rotom.layout<dims = [#rotom.dim<[1:4:1]>, #rotom.dim<[0:4:1]>, #rotom.dim<[R:4:1]>], n = 16, rolls = [(1, 2)]>
#seed_a = #rotom.seed<layouts = [#layout_diag]>
#seed_b = #rotom.seed<layouts = [#layout_b]>

module {
  func.func @main(%a: tensor<4x4xf32> {rotom.seed = #seed_a}, %b: tensor<4x4xf32> {rotom.seed = #seed_b}) -> tensor<4x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %s = arith.addf %a, %a : tensor<4x4xf32>
    %empty = tensor.empty() : tensor<4x4xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x4xf32>) -> tensor<4x4xf32>
    %0 = linalg.matmul ins(%s, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%fill : tensor<4x4xf32>) -> tensor<4x4xf32>
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

  // The diagonalized source stays compact at one ciphertext, and the chosen
  // plan is the slot-diagonal family: its result spans the four j
  // ciphertexts (the summed k' is a slot gap), not the ct-k family's single
  // compacted ciphertext.
  EXPECT_EQ(cast<RankedTensorType>(main.getArgument(0).getType()).getDimSize(0),
            1);
  EXPECT_EQ(
      cast<RankedTensorType>(main.getFunctionType().getResult(0)).getDimSize(0),
      4);

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
        expected[i][j] += 2 * lhs[i][k] * rhs[k][j];
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

// The general-search rolled-candidate hook (single-roll seed variants): %b
// arrives packed ONLY at the diagonal roll(1,0) [0:4];[1:4], while %a is
// seeded plain row-major. Aligning the add at either seed would cost a real
// rolled<->unrolled conversion; instead the search picks %a's own diagonal
// single-roll variant -- the same packing, available at encode time for
// free -- so both operands and the result share the rolled layout and the
// lowered function is ONE add with no rotations, masks, or convert_layouts.
TEST(RotomPipelineExecutionTest, ElementwiseSourceAdoptsDiagonalVariant) {
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = openfhe::parse(&context, R"mlir(
#layout_row = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:4:1]>], n = 16>
#layout_diag = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:4:1]>], n = 16, rolls = [(1, 0)]>
#seed_a = #rotom.seed<layouts = [#layout_row]>
#seed_b = #rotom.seed<layouts = [#layout_diag]>

module {
  func.func @main(%a: tensor<4x4xf32> {rotom.seed = #seed_a}, %b: tensor<4x4xf32> {rotom.seed = #seed_b}) -> tensor<4x4xf32> {
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
  tensor_ext::LayoutAttr resultLayout = getResultLayout(main, 0);
  ASSERT_TRUE(lhsLayout);
  ASSERT_TRUE(rhsLayout);
  ASSERT_TRUE(resultLayout);

  // Both operands sit at the same (diagonal) packing: no conversion ops.
  EXPECT_EQ(lhsLayout, rhsLayout);
  int64_t nonAddOps = 0;
  main.walk([&](Operation* op) {
    if (isa<tensor_ext::ConvertLayoutOp, tensor_ext::RotateOp>(op)) {
      ++nonAddOps;
    }
  });
  EXPECT_EQ(nonAddOps, 0);

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
      expected[i][j] = lhs[i][j] + rhs[i][j];
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
