#include <cstdint>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"  // from @google_benchmark
#include "benchmark/isl/relations.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Transforms/ConvertToCiphertextSemantics/AssignLayout.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"              // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {

static void BM_implementAssignLayout_helper(benchmark::State& state,
                                            llvm::ArrayRef<int64_t> shape,
                                            int64_t ciphertext_size,
                                            const std::string& relation_str) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor_ext::TensorExtDialect,
                      scf::SCFDialect, tensor::TensorDialect>();

  OpBuilder opBuilder(&context);
  Location loc = opBuilder.getUnknownLoc();

  auto inputType = RankedTensorType::get(shape, opBuilder.getF32Type());

  int64_t numElements = inputType.getNumElements();
  std::vector<float> values(numElements);
  for (int64_t i = 0; i < numElements; ++i) {
    values[i] = static_cast<float>(i);
  }
  auto constantAttr =
      DenseElementsAttr::get(inputType, ArrayRef<float>(values));

  auto layoutAttr = tensor_ext::LayoutAttr::get(&context, relation_str);
  CodegenStrategy strategy = static_cast<CodegenStrategy>(state.range(0));

  for (auto _ : state) {
    Block* block = new Block();
    ImplicitLocOpBuilder builder(loc, &context);
    builder.setInsertionPointToEnd(block);

    auto constantOp = arith::ConstantOp::create(builder, loc, constantAttr);
    Value inputVal = constantOp.getResult();

    auto result = implementAssignLayout(
        inputVal, layoutAttr, ciphertext_size, builder, [](Operation* op) {},
        /*domainSchedule=*/{}, strategy);

    benchmark::DoNotOptimize(result);
    delete block;
  }
}

static void BM_implementAssignLayout_Relation1(benchmark::State& state) {
  BM_implementAssignLayout_helper(state, {2, 17, 21}, 8192, kRelation1);
}
BENCHMARK(BM_implementAssignLayout_Relation1)
    ->Arg(static_cast<int64_t>(CodegenStrategy::AUTO))
    ->Arg(static_cast<int64_t>(CodegenStrategy::NEVER_FOLD))
    ->Arg(static_cast<int64_t>(CodegenStrategy::FOLD_WHEN_POSSIBLE))
    ->Unit(benchmark::kSecond);

static void BM_implementAssignLayout_Relation2(benchmark::State& state) {
  BM_implementAssignLayout_helper(state, {16, 22, 2}, 4096, kRelation2);
}
BENCHMARK(BM_implementAssignLayout_Relation2)
    ->Arg(static_cast<int64_t>(CodegenStrategy::AUTO))
    ->Arg(static_cast<int64_t>(CodegenStrategy::NEVER_FOLD))
    ->Arg(static_cast<int64_t>(CodegenStrategy::FOLD_WHEN_POSSIBLE))
    ->Unit(benchmark::kSecond);

static void BM_implementAssignLayout_Relation3(benchmark::State& state) {
  BM_implementAssignLayout_helper(state, {48, 48, 9}, 4096, kRelation3);
}
BENCHMARK(BM_implementAssignLayout_Relation3)
    ->Arg(static_cast<int64_t>(CodegenStrategy::AUTO))
    ->Arg(static_cast<int64_t>(CodegenStrategy::NEVER_FOLD))
    ->Arg(static_cast<int64_t>(CodegenStrategy::FOLD_WHEN_POSSIBLE))
    ->Unit(benchmark::kSecond);

static void BM_implementAssignLayout_HotwordRelationA(benchmark::State& state) {
  BM_implementAssignLayout_helper(state, {16, 40, 3}, 4096, kLayout4Relation);
}
BENCHMARK(BM_implementAssignLayout_HotwordRelationA)
    ->Arg(static_cast<int64_t>(CodegenStrategy::AUTO))
    ->Arg(static_cast<int64_t>(CodegenStrategy::NEVER_FOLD))
    ->Arg(static_cast<int64_t>(CodegenStrategy::FOLD_WHEN_POSSIBLE))
    ->Unit(benchmark::kSecond);

static void BM_implementAssignLayout_HotwordRelationB(benchmark::State& state) {
  BM_implementAssignLayout_helper(state, {24, 16, 1}, 4096, kLayout8Relation);
}
BENCHMARK(BM_implementAssignLayout_HotwordRelationB)
    ->Arg(static_cast<int64_t>(CodegenStrategy::AUTO))
    ->Arg(static_cast<int64_t>(CodegenStrategy::NEVER_FOLD))
    ->Arg(static_cast<int64_t>(CodegenStrategy::FOLD_WHEN_POSSIBLE))
    ->Unit(benchmark::kSecond);

static void BM_implementAssignLayout_HotwordRelationC(benchmark::State& state) {
  BM_implementAssignLayout_helper(state, {1, 24, 49}, 4096, kLayout10Relation);
}
BENCHMARK(BM_implementAssignLayout_HotwordRelationC)
    ->Arg(static_cast<int64_t>(CodegenStrategy::AUTO))
    ->Arg(static_cast<int64_t>(CodegenStrategy::NEVER_FOLD))
    ->Arg(static_cast<int64_t>(CodegenStrategy::FOLD_WHEN_POSSIBLE))
    ->Unit(benchmark::kSecond);

static void BM_implementAssignLayout_HotwordRelationD(benchmark::State& state) {
  BM_implementAssignLayout_helper(state, {24, 16, 9}, 4096, kLayout13Relation);
}
BENCHMARK(BM_implementAssignLayout_HotwordRelationD)
    ->Arg(static_cast<int64_t>(CodegenStrategy::AUTO))
    ->Arg(static_cast<int64_t>(CodegenStrategy::NEVER_FOLD))
    ->Arg(static_cast<int64_t>(CodegenStrategy::FOLD_WHEN_POSSIBLE))
    ->Unit(benchmark::kSecond);

}  // namespace heir
}  // namespace mlir

BENCHMARK_MAIN();
