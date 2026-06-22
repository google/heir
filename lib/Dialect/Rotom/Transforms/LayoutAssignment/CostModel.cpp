#include "lib/Dialect/Rotom/Transforms/LayoutAssignment/CostModel.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>

#include "llvm/include/llvm/ADT/StringRef.h"         // from @llvm-project
#include "llvm/include/llvm/Support/Error.h"         // from @llvm-project
#include "llvm/include/llvm/Support/JSON.h"          // from @llvm-project
#include "llvm/include/llvm/Support/MemoryBuffer.h"  // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"   // from @llvm-project

namespace mlir::heir::rotom {

std::optional<RotomCostModel> parseCostModel(llvm::StringRef json) {
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(json);
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return std::nullopt;
  }
  const llvm::json::Object* obj = parsed->getAsObject();
  if (!obj) return std::nullopt;

  RotomCostModel model;
  if (auto v = obj->getInteger("rotation")) model.rotation = *v;
  if (auto v = obj->getInteger("ciphertextMultiply"))
    model.ciphertextMultiply = *v;
  if (auto v = obj->getInteger("add")) model.add = *v;
  return model;
}

namespace {

// Loads the initial cost model: defaults, overridden by the JSON file named in
// the ROTOM_COST_MODEL environment variable when present and readable.
RotomCostModel loadInitialCostModel() {
  const char* path = std::getenv("ROTOM_COST_MODEL");
  if (!path) return RotomCostModel{};

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    llvm::errs() << "rotom: cannot read ROTOM_COST_MODEL '" << path
                 << "', using default cost model\n";
    return RotomCostModel{};
  }
  if (std::optional<RotomCostModel> model =
          parseCostModel((*buffer)->getBuffer())) {
    return *model;
  }
  llvm::errs() << "rotom: malformed ROTOM_COST_MODEL '" << path
               << "', using default cost model\n";
  return RotomCostModel{};
}

RotomCostModel& mutableCostModel() {
  static RotomCostModel model = loadInitialCostModel();
  return model;
}

}  // namespace

const RotomCostModel& getCostModel() { return mutableCostModel(); }

void setCostModel(const RotomCostModel& model) { mutableCostModel() = model; }

}  // namespace mlir::heir::rotom
