#include "lib/Utils/TransformUtils.h"

#include <cassert>
#include <numeric>
#include <set>
#include <string>
#include <string_view>

#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

#define DEBUG_TYPE "transform-utils"

namespace mlir {
namespace heir {

static constexpr int kUnset = -1;

func::FuncOp detectEntryFunction(ModuleOp moduleOp,
                                 std::string_view entryFunction) {
  // get from user input
  auto entryFunc = moduleOp.lookupSymbol<func::FuncOp>(entryFunction);
  if (!entryFunc) {
    // detect the entry function with the following heuristic:
    // 1. the function name does not contain "__"
    // 2. the function is not a declaration
    // 3. the function is not called by any other function
    // 4. the first function that satisfies the above conditions

    // get all the called functions
    std::set<std::string> calledFuncs;
    moduleOp->walk<WalkOrder::PreOrder>([&](func::CallOp callOp) {
      auto callee = callOp.getCallee();
      calledFuncs.insert(std::string(callee));
    });

    moduleOp->walk<WalkOrder::PreOrder>([&](func::FuncOp funcOp) {
      auto funcSymName = funcOp.getSymName();
      if (funcSymName.find("__") != std::string::npos ||
          calledFuncs.find(std::string(funcSymName)) != calledFuncs.end() ||
          funcOp.isDeclaration()) {
        return WalkResult::advance();
      }
      entryFunc = funcOp;
      return WalkResult::interrupt();
    });
  }
  // still no result then emit warning
  if (!entryFunc) {
    moduleOp->emitWarning(
        "Entry function not found, please provide entry-function in the pass "
        "options");
  }
  return entryFunc;
}

Value convertIntegerValueToTensorOfBits(Value integer, OpBuilder &b,
                                        Location loc) {
  IntegerType argType = mlir::cast<IntegerType>(integer.getType());
  int width = argType.getWidth();
  if (width == 1) {
    return integer;
  }

  SmallVector<Value> insertValues;
  for (int i = 0; i < width; i++) {
    // These arith ops correspond to extracting the i-th bit
    // from the input
    auto shiftAmount =
        b.create<arith::ConstantOp>(loc, argType, b.getIntegerAttr(argType, i));
    auto bitMask = b.create<arith::ConstantOp>(
        loc, argType, b.getIntegerAttr(argType, 1 << i));
    auto andOp = b.create<arith::AndIOp>(loc, integer, bitMask);
    auto shifted = b.create<arith::ShRSIOp>(loc, andOp, shiftAmount);
    insertValues.push_back(
        b.create<arith::TruncIOp>(loc, b.getI1Type(), shifted));
  }

  auto result = b.create<tensor::FromElementsOp>(loc, insertValues);
  return result;
}

Value convertTensorOfBitsToInteger(Value tensor, Type resultType, OpBuilder &b,
                                   Location loc) {
  auto tensorType = cast<TensorType>(tensor.getType());
  auto integerType = cast<IntegerType>(resultType);
  assert(tensorType.getRank() == 1 && "Expected tensor of bits to be 1D");

  Value result =
      b.create<arith::ConstantIntOp>(loc, integerType, 0).getResult();
  for (int i = 0; i < tensorType.getNumElements(); i++) {
    // The i-th bit of the tensor is stored at bit position i
    auto loadOp = b.create<tensor::ExtractOp>(
        loc, tensor, ValueRange{b.create<arith::ConstantIndexOp>(loc, i)});
    auto extOp = b.create<arith::ExtSIOp>(loc, integerType, loadOp.getResult());
    auto shiftAmount = b.create<arith::ConstantIntOp>(loc, integerType, i);
    auto shifted = b.create<arith::ShLIOp>(loc, extOp, shiftAmount);
    auto orOp = b.create<arith::OrIOp>(loc, integerType, result, shifted);
    result = orOp.getResult();
  }

  return result;
}

int64_t getMinUnusedTarget(llvm::ArrayRef<int64_t> perm) {
  std::vector<int64_t> unmappedOutputsVector(perm.size());
  std::iota(unmappedOutputsVector.begin(), unmappedOutputsVector.end(), 0);
  std::set<int64_t> unmappedOutputs(unmappedOutputsVector.begin(),
                                    unmappedOutputsVector.end());
  for (int64_t target : perm) {
    if (target != kUnset) {
      unmappedOutputs.erase(target);
    }
  }

  if (unmappedOutputs.empty()) {
    return -1;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Unmapped outputs: ";
    for (int64_t i : unmappedOutputs) {
      llvm::dbgs() << i << " ";
    }
    llvm::dbgs() << "\n";
  });

  return *unmappedOutputs.begin();
}

int64_t getMinUnusedInput(llvm::ArrayRef<int64_t> perm) {
  for (int64_t i = 0; i < perm.size(); ++i) {
    if (perm[i] == kUnset) return i;
  }
  return -1;
}

}  // namespace heir
}  // namespace mlir
