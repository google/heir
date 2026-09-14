#include "lib/Transforms/LoweringHistory/LoweringHistory.h"

#include "llvm/include/llvm/ADT/DenseSet.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"           // from @llvm-project

namespace mlir::heir {
#define GEN_PASS_DEF_RECORDLOWERINGHISTORY
#define GEN_PASS_DEF_EXPLAINLOWERINGHISTORY
#include "lib/Transforms/LoweringHistory/LoweringHistory.h.inc"

namespace {
constexpr llvm::StringLiteral kStage = "heir.lowering_stage";
constexpr llvm::StringLiteral kPass = "heir.lowering_pass";
constexpr llvm::StringLiteral kResult = "heir.lowering_result";
constexpr llvm::StringLiteral kNotePrefix = "lowering history: ";

struct RecordLoweringHistory
    : impl::RecordLoweringHistoryBase<RecordLoweringHistory> {
  using RecordLoweringHistoryBase::RecordLoweringHistoryBase;

  void runOnOperation() override {
    if (stage.empty()) {
      getOperation()->emitError(
          "record-lowering-history requires a nonempty stage");
      return signalPassFailure();
    }
    auto* context = &getContext();
    auto metadata = DictionaryAttr::get(
        context, {NamedAttribute(StringAttr::get(context, kStage),
                                 StringAttr::get(context, stage))});
    getOperation()->walk([&](Operation* op) {
      auto named = NameLoc::get(op->getName().getIdentifier(), op->getLoc());
      // The NameLoc separates nested checkpoints, including repeated stages,
      // so FusedLoc's normalization cannot flatten distinct history entries.
      op->setLoc(FusedLoc::get(context, {named}, metadata));
    });
    markAllAnalysesPreserved();
  }
};

struct ExplainLoweringHistory
    : impl::ExplainLoweringHistoryBase<ExplainLoweringHistory> {
  using ExplainLoweringHistoryBase::ExplainLoweringHistoryBase;

  void runOnOperation() override {
    // Capture the option by value: the handler outlives this pass and belongs
    // to the context. Do not retain a pass or operation pointer.
    unsigned limit = maxNotes;
    getContext().getDiagEngine().registerHandler([limit](Diagnostic& diag) {
      if (diag.getSeverity() != DiagnosticSeverity::Error || limit == 0)
        return failure();
      // Enabling the handler again (e.g. in a nested pipeline) must not append
      // the same explanation twice. Other notes are left intact.
      for (auto& note : diag.getNotes()) {
        if (StringRef(note.str()).starts_with(kNotePrefix)) return failure();
      }
      unsigned count = 0;
      llvm::DenseSet<Location> visited;
      diag.getLocation()->walk([&](Location loc) {
        if (!visited.insert(loc).second) return WalkResult::skip();
        auto fused = dyn_cast<FusedLoc>(loc);
        if (!fused) return WalkResult::advance();
        auto metadata = dyn_cast_or_null<DictionaryAttr>(fused.getMetadata());
        if (!metadata) return WalkResult::advance();
        auto stage = metadata.getAs<StringAttr>(kStage);
        auto pass = metadata.getAs<StringAttr>(kPass);
        auto result = metadata.getAs<StringAttr>(kResult);
        if ((!stage && !(pass && result)) || fused.getLocations().size() != 1)
          return WalkResult::advance();
        auto named = dyn_cast<NameLoc>(fused.getLocations().front());
        if (!named) return WalkResult::advance();
        if (count == limit) {
          diag.attachNote() << kNotePrefix << "additional checkpoints omitted";
          return WalkResult::interrupt();
        }
        ++count;
        auto& note = diag.attachNote(named.getChildLoc());
        note << kNotePrefix << "'" << named.getName().getValue();
        if (pass && result)
          note << "' lowered to '" << result.getValue() << "' by --"
               << pass.getValue();
        else
          note << "' observed at stage '" << stage.getValue() << "'";
        return WalkResult::advance();
      });
      // Let the normal handler print/verify the augmented diagnostic.
      return failure();
    });
    markAllAnalysesPreserved();
  }
};
}  // namespace
Location getLoweringLocation(Operation* source, StringRef pass,
                             StringRef resultOperation) {
  bool enabled = false;
  source->getLoc()->walk([&](Location loc) {
    auto fused = dyn_cast<FusedLoc>(loc);
    if (!fused) return WalkResult::advance();
    auto metadata = dyn_cast_or_null<DictionaryAttr>(fused.getMetadata());
    if (metadata && (metadata.getAs<StringAttr>(kStage) ||
                     metadata.getAs<StringAttr>(kPass))) {
      enabled = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!enabled) return source->getLoc();
  auto* context = source->getContext();
  auto metadata = DictionaryAttr::get(
      context, {NamedAttribute(StringAttr::get(context, kPass),
                               StringAttr::get(context, pass)),
                NamedAttribute(StringAttr::get(context, kResult),
                               StringAttr::get(context, resultOperation))});
  auto named =
      NameLoc::get(source->getName().getIdentifier(), source->getLoc());
  return FusedLoc::get(context, {named}, metadata);
}
}  // namespace mlir::heir
