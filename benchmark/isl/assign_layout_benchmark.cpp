#include <cstdint>
#include <vector>

#include "benchmark/benchmark.h"  // from @google_benchmark
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Transforms/ConvertToCiphertextSemantics/AssignLayout.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project

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

  for (auto _ : state) {
    Block* block = new Block();
    ImplicitLocOpBuilder builder(loc, &context);
    builder.setInsertionPointToEnd(block);

    auto constantOp = arith::ConstantOp::create(builder, loc, constantAttr);
    Value inputVal = constantOp.getResult();

    auto result = implementAssignLayout(inputVal, layoutAttr, ciphertext_size,
                                        builder, [](Operation* op) {});

    benchmark::DoNotOptimize(result);
    delete block;
  }
}

static void BM_implementAssignLayout_Relation1(benchmark::State& state) {
  BM_implementAssignLayout_helper(
      state, {2, 17, 21}, 8192,
      "{ [i0, i1, i2] -> [ct, slot] : ct = 0 and (357i0 + 84i1 + 272i2 + "
      "slot) mod 714 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 16 and 0 <= i2 <= "
      "20 and 0 <= slot <= 8191 }");
}
BENCHMARK(BM_implementAssignLayout_Relation1)->Unit(benchmark::kSecond);

static void BM_implementAssignLayout_Relation2(benchmark::State& state) {
  BM_implementAssignLayout_helper(
      state, {16, 22, 2}, 4096,
      "{ [i0, i1, i2] -> [ct, slot] : (2048 + 98i0 - 100i1 - i2 + ct + "
      "2048*floor((-98 - 98i0 + slot)/2048)) mod 4096 = 0 and 0 <= i0 <= 15 "
      "and 0 <= i1 <= 21 and 0 <= i2 <= 1 and 0 <= ct <= 2047 and 0 <= slot "
      "<= 2146 and 2048*floor((-98 - 98i0 + slot)/2048) >= -3615 + slot and "
      "2048*floor((-98 - 98i0 + slot)/2048) >= -2147 - 98i0 + i2 + slot and "
      "2048*floor((-98 - 98i0 + slot)/2048) <= -2048 - 98i0 + slot and "
      "2048*floor((-98 - 98i0 + slot)/2048) <= -2048 - 98i0 + i2 + slot and "
      "2048*floor((-98 - 98i0 + slot)/2048) <= -2048 + slot }");
}
BENCHMARK(BM_implementAssignLayout_Relation2)->Unit(benchmark::kSecond);

static void BM_implementAssignLayout_Relation3(benchmark::State& state) {
  BM_implementAssignLayout_helper(
      state, {48, 48, 9}, 4096,
      "{ [d0, d1, d2] -> [r0, r1] : exists (l0, l1 : -13*d0 + 21*d1 + d2 - "
      "r0 + 1024*l1 = 0 and 0 <= d0 <= 47 and 0 <= d1 <= 47 and 0 <= d2 <= "
      "8 and 0 <= r0 <= 1023 and 0 <= r1 <= 4095 and 13*d0 - r1 + 1024*l0 + "
      "1036 >= 0 and -13*d0 + r1 - 1024*l0 - 1024 >= 0 and r1 - 1024*l0 - "
      "1024 >= 0) }");
}
BENCHMARK(BM_implementAssignLayout_Relation3)->Unit(benchmark::kSecond);

}  // namespace heir
}  // namespace mlir

BENCHMARK_MAIN();
