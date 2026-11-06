#include "lib/Bindings/Python/HEIRCAPI.h"

#include <cstdint>
#include <memory>
#include <mutex>  // NOLINT(build/c++11)
#include <string>
#include <vector>

#include "lib/Tools/HeirOptRegistration.h"
#include "lib/Tools/HeirTranslateRegistration.h"
#include "llvm/include/llvm/ADT/STLExtras.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/ScopeExit.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/Twine.h"                // from @llvm-project
#include "llvm/include/llvm/Support/CommandLine.h"      // from @llvm-project
#include "llvm/include/llvm/Support/MemoryBuffer.h"     // from @llvm-project
#include "llvm/include/llvm/Support/SourceMgr.h"        // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"      // from @llvm-project
#include "mlir/include/mlir/Bytecode/BytecodeWriter.h"  // from @llvm-project
#include "mlir/include/mlir/CAPI/IR.h"                  // from @llvm-project
#include "mlir/include/mlir/CAPI/Support.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"           // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"       // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Verifier.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project

namespace {

using TranslationOpt =
    llvm::cl::opt<const mlir::Translation*, false, mlir::TranslationParser>;

constexpr const char* kTranslationOptName = "heir-capi-translation";

void ensureTranslationsRegistered() {
  static std::once_flag flag;
  std::call_once(flag, [] { mlir::heir::registerAllHEIRTranslations(); });
}

// The translation registry is only accessible through a command line parser,
// which snapshots the registry when it is constructed. Construct it lazily
// (after the translations are registered) and never destroy it, since the
// global command line registry keeps a pointer to it.
TranslationOpt& getTranslationOpt() {
  ensureTranslationsRegistered();
  static TranslationOpt* opt = new TranslationOpt(
      llvm::StringRef(kTranslationOptName),
      llvm::cl::desc("Translation used by heirTranslate"), llvm::cl::Hidden);
  return *opt;
}

std::mutex& getTranslateMutex() {
  static std::mutex mutex;
  return mutex;
}

// Extracts "foo" from "--foo=bar", "-foo", or "--foo".
llvm::StringRef optionName(llvm::StringRef arg) {
  arg = arg.ltrim('-');
  return arg.take_until([](char c) { return c == '='; });
}

// Built-in llvm::cl flags that print to stdout/stderr or call std::exit()
// inside ParseCommandLineOptions (`-h` is an alias of `--help`).
bool isDisallowedCLOption(llvm::StringRef name) {
  return name == "help" || name == "h" || name.starts_with("help-") ||
         name == "version" || name == "print-options" ||
         name == "print-all-options";
}

void emitError(MlirStringCallback callback, void* userData,
               llvm::StringRef message) {
  callback(wrap(message), userData);
}

}  // namespace

void heirRegisterAllDialects(MlirDialectRegistry registry) {
  mlir::heir::registerAllHEIRDialects(*unwrap(registry));
}

void heirRegisterAllPasses(void) {
  static std::once_flag flag;
  std::call_once(flag, [] { mlir::heir::registerAllHEIRPasses(); });
}

void heirForEachTranslation(HeirTranslationCallback callback, void* userData) {
  std::lock_guard<std::mutex> lock(getTranslateMutex());
  TranslationOpt& opt = getTranslationOpt();
  auto& parser = opt.getParser();
  for (unsigned i = 0, e = parser.getNumOptions(); i < e; ++i) {
    callback(wrap(parser.getOption(i)), wrap(parser.getDescription(i)),
             userData);
  }
}

MlirLogicalResult heirTranslate(MlirOperation op, MlirStringRef translation,
                                intptr_t numOptions,
                                const MlirStringRef* options,
                                MlirStringCallback outputCallback,
                                void* outputUserData,
                                MlirStringCallback errorCallback,
                                void* errorUserData) {
  std::lock_guard<std::mutex> lock(getTranslateMutex());
  TranslationOpt& opt = getTranslationOpt();

  // Validate the translation name up front: errors for unknown option values
  // are printed to stderr rather than to the error stream given to the parser.
  auto& parser = opt.getParser();
  if (parser.findOption(unwrap(translation)) == parser.getNumOptions()) {
    emitError(errorCallback, errorUserData,
              ("unknown translation '" + unwrap(translation) + "'").str());
    return mlirLogicalResultFailure();
  }

  // Select the translation and set its options by parsing a synthetic
  // heir-translate command line.
  std::vector<std::string> args = {
      "heir-translate",
      ("--" + llvm::Twine(kTranslationOptName) + "=" + unwrap(translation))
          .str()};
  auto& registered = llvm::cl::getRegisteredOptions();
  for (intptr_t i = 0; i < numOptions; ++i) {
    llvm::StringRef optStr = unwrap(options[i]);
    llvm::StringRef name = optionName(optStr);
    // Only accept registered options (e.g., translation options), so that
    // unknown options get a clear error (parse errors for them would only
    // name the synthetic command line).
    if (!optStr.starts_with("-") || !registered.contains(name)) {
      emitError(errorCallback, errorUserData,
                ("unknown option '" + optStr + "'").str());
      return mlirLogicalResultFailure();
    }
    if (isDisallowedCLOption(name)) {
      emitError(errorCallback, errorUserData,
                ("unsupported option '" + optStr + "'").str());
      return mlirLogicalResultFailure();
    }
    args.push_back(optStr.str());
  }
  std::vector<const char*> argv;
  argv.reserve(args.size());
  for (const std::string& arg : args) argv.push_back(arg.c_str());

  // Options are global state: restore every option we touch to its default
  // afterwards, so that one call's options don't leak into the next.
  llvm::scope_exit resetOptions([&] {
    for (const std::string& arg : args) {
      auto it = registered.find(optionName(arg));
      if (it != registered.end()) it->second->setDefault();
    }
    llvm::cl::ResetAllOptionOccurrences();
  });

  std::string parseErrors;
  llvm::raw_string_ostream parseErrorStream(parseErrors);
  llvm::cl::ResetAllOptionOccurrences();
  if (!llvm::cl::ParseCommandLineOptions(argv.size(), argv.data(), "",
                                         &parseErrorStream)) {
    if (parseErrors.empty()) {
      // Errors about invalid option values (e.g., an unknown enum value or a
      // missing value) are printed to stderr rather than to parseErrorStream.
      parseErrors = "invalid translation options";
      for (const std::string& arg : llvm::drop_begin(args, 2))
        parseErrors += " " + arg;
      parseErrors += " (see stderr for details)";
    }
    emitError(errorCallback, errorUserData, parseErrors);
    return mlirLogicalResultFailure();
  }
  const mlir::Translation* translate = opt.getValue();
  if (!translate) {
    emitError(errorCallback, errorUserData,
              ("unknown translation: " + unwrap(translation)).str());
    return mlirLogicalResultFailure();
  }

  mlir::Operation* operation = unwrap(op);
  mlir::MLIRContext* context = operation->getContext();

  std::string diagnostics;
  llvm::raw_string_ostream diagnosticStream(diagnostics);
  mlir::ScopedDiagnosticHandler handler(context, [&](mlir::Diagnostic& diag) {
    diagnosticStream << diag.getLocation() << ": " << diag << "\n";
    for (mlir::Diagnostic& note : diag.getNotes())
      diagnosticStream << note.getLocation() << ": note: " << note << "\n";
    return mlir::success();
  });

  if (mlir::failed(mlir::verify(operation))) {
    emitError(
        errorCallback, errorUserData,
        diagnostics.empty() ? "operation failed verification" : diagnostics);
    return mlirLogicalResultFailure();
  }

  // Translations operate on a source buffer (which they parse and verify, and
  // are allowed to mutate), so hand them a bytecode copy of the op.
  std::string bytecode;
  llvm::raw_string_ostream bytecodeStream(bytecode);
  if (mlir::failed(mlir::writeBytecodeToFile(operation, bytecodeStream))) {
    emitError(errorCallback, errorUserData,
              diagnostics.empty()
                  ? "failed to serialize the operation to bytecode"
                  : diagnostics);
    return mlirLogicalResultFailure();
  }
  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  sourceMgr->AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(bytecode, "<heir-python>"),
      llvm::SMLoc());

  std::string output;
  llvm::raw_string_ostream outputStream(output);
  if (mlir::failed((*translate)(sourceMgr, outputStream, context))) {
    emitError(errorCallback, errorUserData,
              diagnostics.empty() ? "translation failed" : diagnostics);
    return mlirLogicalResultFailure();
  }
  outputCallback(wrap(llvm::StringRef(output)), outputUserData);
  return mlirLogicalResultSuccess();
}
