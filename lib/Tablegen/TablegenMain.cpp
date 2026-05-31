#include "lib/Tablegen/CompilationTargetEmitter.h"
#include "llvm/include/llvm/Support/CommandLine.h"  // from @llvm-project
#include "llvm/include/llvm/Support/InitLLVM.h"     // from @llvm-project
#include "llvm/include/llvm/TableGen/Main.h"        // from @llvm-project
#include "llvm/include/llvm/TableGen/Record.h"      // from @llvm-project

using namespace mlir;
using namespace heir;

enum ActionType {
  None,
  GenCompilationTargetRegistration,
};

static llvm::cl::opt<ActionType> action(
    llvm::cl::desc("Action to perform:"),
    llvm::cl::values(clEnumValN(GenCompilationTargetRegistration,
                                "gen-compilation-target-registration",
                                "Generate compilation target registration")));

bool heirTableGenMain(llvm::raw_ostream& os,
                      const llvm::RecordKeeper& records) {
  switch (action) {
    case GenCompilationTargetRegistration:
      return emitCompilationTargetRegistration(records, os);
    default:
      return false;
  }
}

int main(int argc, char** argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  return llvm::TableGenMain(argv[0], &heirTableGenMain);
}
