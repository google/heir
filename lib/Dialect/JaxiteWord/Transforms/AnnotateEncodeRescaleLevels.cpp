#include "lib/Dialect/JaxiteWord/Transforms/AnnotateEncodeRescaleLevels.h"

#include <optional>

#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordDialect.h"
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordOps.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingOps.h"
#include "llvm/include/llvm/ADT/DenseMap.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"          // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir {
namespace heir {
namespace jaxiteword {

#define GEN_PASS_DEF_ANNOTATEENCODERESCALELEVELS
#include "lib/Dialect/JaxiteWord/Transforms/Passes.h.inc"

namespace {

std::optional<int64_t> getCiphertextCurrentLevel(Value value) {
  auto type = dyn_cast<lwe::LWECiphertextType>(value.getType());
  if (!type) return std::nullopt;
  return type.getModulusChain().getCurrent();
}

bool hasDownstreamModReduce(Value value) {
  auto initialLevel = getCiphertextCurrentLevel(value);
  if (!initialLevel) return false;

  DenseSet<Value> visited;
  SmallVector<Value> worklist{value};
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second) continue;
    for (Operation* user : current.getUsers()) {
      if (isa<ckks::RescaleOp, lwe::ModDownOp, jaxiteword::ModReduceOp>(user))
        return true;
      for (Value result : user->getResults()) {
        auto resultLevel = getCiphertextCurrentLevel(result);
        if (resultLevel && resultLevel.value() == initialLevel.value()) {
          worklist.push_back(result);
        }
      }
    }
  }
  return false;
}

std::optional<int64_t> getLweMulPlainLevel(lwe::RMulPlainOp op,
                                           Value plaintext) {
  if (!hasDownstreamModReduce(op.getOutput())) return std::nullopt;
  Value other = op.getLhs() == plaintext ? op.getRhs() : op.getLhs();
  return getCiphertextCurrentLevel(other);
}

std::optional<int64_t> getJaxiteWordMulPlainLevel(jaxiteword::MulPlainOp op,
                                                  Value plaintext) {
  if (op.getPlaintext() != plaintext) return std::nullopt;
  if (!hasDownstreamModReduce(op.getOutput())) return std::nullopt;
  return getCiphertextCurrentLevel(op.getCiphertext());
}

std::optional<int64_t> getMulPlainLevel(Value plaintext) {
  std::optional<int64_t> inferredLevel;
  for (Operation* user : plaintext.getUsers()) {
    std::optional<int64_t> userLevel;
    if (auto mulPlain = dyn_cast<lwe::RMulPlainOp>(user)) {
      userLevel = getLweMulPlainLevel(mulPlain, plaintext);
    } else if (auto mulPlain = dyn_cast<jaxiteword::MulPlainOp>(user)) {
      userLevel = getJaxiteWordMulPlainLevel(mulPlain, plaintext);
    }
    if (!userLevel) continue;

    if (inferredLevel && inferredLevel.value() != userLevel.value()) {
      return std::nullopt;
    }
    inferredLevel = userLevel;
  }
  return inferredLevel;
}

template <typename EncodeOp>
FailureOr<std::optional<int64_t>> inferRescaleLevel(
    EncodeOp op, const DenseMap<uint32_t, int64_t>& levelsBySite) {
  std::optional<int64_t> inferredLevel = getMulPlainLevel(op.getResult());
  for (Operation* user : op.getResult().getUsers()) {
    auto store = dyn_cast<preprocessing::StoreOp>(user);
    if (!store) continue;
    auto levelIt = levelsBySite.find(store.getSiteId());
    if (levelIt == levelsBySite.end()) continue;
    if (inferredLevel && inferredLevel.value() != levelIt->second) {
      return op.emitOpError()
             << "feeds ciphertext-plaintext multiplications at multiple "
                "rescale levels; clone the encode before annotating";
    }
    inferredLevel = levelIt->second;
  }
  return inferredLevel;
}

LogicalResult annotateLweEncode(
    lwe::RLWEEncodeOp op, const DenseMap<uint32_t, int64_t>& levelsBySite) {
  FailureOr<std::optional<int64_t>> inferredLevel =
      inferRescaleLevel(op, levelsBySite);
  if (failed(inferredLevel)) return failure();
  if (!*inferredLevel) return success();
  op.setLevel(**inferredLevel);
  return success();
}

LogicalResult annotateJaxiteWordEncode(
    jaxiteword::EncodeOp op, const DenseMap<uint32_t, int64_t>& levelsBySite) {
  FailureOr<std::optional<int64_t>> inferredLevel =
      inferRescaleLevel(op, levelsBySite);
  if (failed(inferredLevel)) return failure();
  if (!*inferredLevel) return success();
  auto attr =
      IntegerAttr::get(IntegerType::get(op.getContext(), 64), **inferredLevel);
  op.setRescaleLevelAttr(attr);
  return success();
}

struct AnnotateEncodeRescaleLevels
    : impl::AnnotateEncodeRescaleLevelsBase<AnnotateEncodeRescaleLevels> {
  using AnnotateEncodeRescaleLevelsBase::AnnotateEncodeRescaleLevelsBase;

  void runOnOperation() override {
    ModuleOp module = cast<ModuleOp>(getOperation());
    DenseMap<uint32_t, int64_t> levelsBySite;
    bool hasConflictingSite = false;
    module.walk(
        [&](preprocessing::LoadOp load) {
          auto level = getMulPlainLevel(load.getResult());
          if (!level) return;
          auto [it, inserted] =
              levelsBySite.try_emplace(load.getSiteId(), level.value());
          if (!inserted && it->second != level.value()) {
            load.emitOpError()
                << "preprocessing site " << load.getSiteId()
                << " feeds ciphertext-plaintext multiplications at multiple "
                   "rescale levels";
            hasConflictingSite = true;
          }
        });
    if (hasConflictingSite) {
      signalPassFailure();
      return;
    }

    WalkResult result = module.walk([&](Operation* op) {
      if (auto encode = dyn_cast<lwe::RLWEEncodeOp>(op)) {
        if (failed(annotateLweEncode(encode, levelsBySite)))
          return WalkResult::interrupt();
      } else if (auto encode = dyn_cast<jaxiteword::EncodeOp>(op)) {
        if (failed(annotateJaxiteWordEncode(encode, levelsBySite)))
          return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) signalPassFailure();
  }
};

}  // namespace
}  // namespace jaxiteword
}  // namespace heir
}  // namespace mlir
