#include "lib/Utils/TransformUtils.h"

#include <numeric>
#include <set>
#include <string>
#include <string_view>

#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project

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

Value convertIntegerValueToMemrefOfBits(Value integer, OpBuilder &b,
                                        Location loc) {
  IntegerType argType = mlir::cast<IntegerType>(integer.getType());
  int width = argType.getWidth();
  if (width == 1) {
    return integer;
  }

  auto allocOp =
      memref::AllocOp::create(b, loc, MemRefType::get({width}, b.getI1Type()));
  for (int i = 0; i < width; i++) {
    // These arith ops correspond to extracting the i-th bit
    // from the input
    auto shiftAmount = arith::ConstantOp::create(b, loc, argType,
                                                 b.getIntegerAttr(argType, i));
    auto bitMask = arith::ConstantOp::create(b, loc, argType,
                                             b.getIntegerAttr(argType, 1 << i));
    auto andOp = arith::AndIOp::create(b, loc, integer, bitMask);
    auto shifted = arith::ShRSIOp::create(b, loc, andOp, shiftAmount);
    memref::StoreOp::create(
        b, loc, arith::TruncIOp::create(b, loc, b.getI1Type(), shifted),
        allocOp, ValueRange{arith::ConstantIndexOp::create(b, loc, i)});
  }

  return allocOp.getResult();
}

Value convertMemrefOfBitsToInteger(Value memref, Type resultType, OpBuilder &b,
                                   Location loc) {
  auto memrefType = cast<MemRefType>(memref.getType());
  auto integerType = cast<IntegerType>(resultType);
  assert(memrefType.getRank() == 1 && "Expected memref of bits to be 1D");

  Value result =
      arith::ConstantIntOp::create(b, loc, integerType, 0).getResult();
  for (int i = 0; i < memrefType.getNumElements(); i++) {
    // The i-th bit of the memref is stored at bit position i
    auto loadOp = memref::LoadOp::create(
        b, loc, memref, ValueRange{arith::ConstantIndexOp::create(b, loc, i)});
    auto extOp =
        arith::ExtSIOp::create(b, loc, integerType, loadOp.getResult());
    auto shiftAmount = arith::ConstantIntOp::create(b, loc, integerType, i);
    auto shifted = arith::ShLIOp::create(b, loc, extOp, shiftAmount);
    auto orOp = arith::OrIOp::create(b, loc, integerType, result, shifted);
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
