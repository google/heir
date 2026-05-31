#include "lib/Tablegen/CompilationTargetEmitter.h"

#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "llvm/include/llvm/TableGen/Record.h"           // from @llvm-project
#include "llvm/include/llvm/TableGen/TableGenBackend.h"  // from @llvm-project

namespace mlir {
namespace heir {

bool emitCompilationTargetRegistration(const llvm::RecordKeeper& records,
                                       llvm::raw_ostream& os) {
  auto targets = records.getAllDerivedDefinitions("CompilationTarget");

  os << "CompilationTargetRegistry::CompilationTargetRegistry() {\n";
  for (auto* target : targets) {
    auto backendName = target->getValueAsString("backendName");
    auto bootstrapLevelsConsumed =
        target->getValueAsInt("bootstrapLevelsConsumed");

    os << "  targets[\"" << backendName << "\"] = CompilationTarget{\""
       << backendName << "\", " << (int)bootstrapLevelsConsumed << "};\n";
  }
  os << "}\n";
  return false;
}

}  // namespace heir
}  // namespace mlir