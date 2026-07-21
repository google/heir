#include "lib/Tablegen/CompilationTargetEmitter.h"
#include "llvm/include/llvm/Support/CommandLine.h"  // from @llvm-project
#include "llvm/include/llvm/Support/InitLLVM.h"     // from @llvm-project
#include "llvm/include/llvm/TableGen/Main.h"        // from @llvm-project
#include "llvm/include/llvm/TableGen/Record.h"      // from @llvm-project

namespace mlir {
namespace heir {

enum ActionType {
  None,
  GenCompilationTargetRegistration,
  GenCompilationTargetOverrides,
  GenCompilationTargetStruct,
};

static llvm::cl::opt<ActionType> action(
    llvm::cl::desc("Action to perform:"),
    llvm::cl::values(clEnumValN(GenCompilationTargetRegistration,
                                "gen-compilation-target-registration",
                                "Generate compilation target registration"),
                     clEnumValN(GenCompilationTargetOverrides,
                                "gen-compilation-target-overrides",
                                "Generate compilation target overrides"),
                     clEnumValN(GenCompilationTargetStruct,
                                "gen-compilation-target-struct",
                                "Generate compilation target struct")));

bool heirTableGenMain(llvm::raw_ostream& os,
                      const llvm::RecordKeeper& records) {
  switch (action) {
    case GenCompilationTargetRegistration:
      return emitCompilationTargetRegistration(records, os);
    case GenCompilationTargetOverrides:
      return emitCompilationTargetOverrides(records, os);
    case GenCompilationTargetStruct:
      return emitCompilationTargetStruct(records, os);
    default:
      return false;
  }
}

}  // namespace heir
}  // namespace mlir

int main(int argc, char** argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  return llvm::TableGenMain(argv[0], &mlir::heir::heirTableGenMain);
}
