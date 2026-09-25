#include "lib/Tablegen/CompilationTargetEmitter.h"

#include <string>
#include <string_view>

#include "llvm/include/llvm/ADT/StringRef.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"           // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"    // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "llvm/include/llvm/TableGen/Record.h"           // from @llvm-project
#include "llvm/include/llvm/TableGen/TableGenBackend.h"  // from @llvm-project

namespace mlir {
namespace heir {

// clang-format off
constexpr std::string_view kOverridePrelude = R"cpp(
#include "mlir/include/mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {

mlir::LogicalResult applyCompilationTargetOverride(CompilationTarget& target,
                                                   StringRef key,
                                                   Attribute value,
                                                   Location loc) {
)cpp";

constexpr std::string_view kIntFieldOverride = R"cpp(
  if (key == "{0}") {{
    auto attr = dyn_cast<IntegerAttr>(value);
    if (!attr) {{ return emitError(loc, "expected integer attribute for key: ") << key; }
    target.{0} = attr.getInt();
    return success();
  }
)cpp";

constexpr std::string_view kStringFieldOverride = R"cpp(
  if (key == "{0}") {{
    auto attr = dyn_cast<StringAttr>(value);
    if (!attr) {{
      return emitError(loc, "expected string attribute for key: ") << key;
    }
    target.{0} = attr.getValue().str();
    return success();
  }
)cpp";

constexpr std::string_view kParserPrelude = R"cpp(
FailureOr<Attribute> parseCompilationTargetOverrideValue(MLIRContext* ctx,
                                                         StringRef key,
                                                         StringRef value,
                                                         Location loc) {
)cpp";

constexpr std::string_view kIntFieldParser = R"cpp(
  if (key == "{0}") {{
    int64_t intVal;
    if (value.getAsInteger(10, intVal)) {{
      return emitError(loc, "expected base-10 integer attribute for key: ") << key;
    }
    return Attribute(IntegerAttr::get(IntegerType::get(ctx, 64), intVal));
  }
)cpp";

constexpr std::string_view kStringFieldParser = R"cpp(
  if (key == "{0}") {{
    return Attribute(StringAttr::get(ctx, value));
  }
)cpp";
// clang-format on

bool emitCompilationTargetRegistration(const llvm::RecordKeeper& records,
                                       llvm::raw_ostream& os) {
  auto targets = records.getAllDerivedDefinitions("CompilationTarget");

  for (auto* target : targets) {
    auto name = target->getName();
    auto bootstrapLevelsConsumed =
        target->getValueAsInt("bootstrapLevelsConsumed");
    auto hasKernelChebyshev = target->getValueAsInt("has_kernel_chebyshev");
    auto hasKernelLinearTransform =
        target->getValueAsInt("has_kernel_linear_transform");
    auto requiresMatchingCiphertextPlaintextLevels =
        target->getValueAsInt("requires_matching_ciphertext_plaintext_levels");
    auto canEmitAdjustScale = target->getValueAsInt("can_emit_adjust_scale");

    os << "void registerTarget" << name << "() {\n"
       << "  "
          "CompilationTargetRegistry::registerTarget(CompilationTarget{"
          "BackendName::"
       << name << ", " << bootstrapLevelsConsumed << ", " << hasKernelChebyshev
       << ", " << hasKernelLinearTransform << ", "
       << requiresMatchingCiphertextPlaintextLevels << ", "
       << canEmitAdjustScale << "});\n"
       << "}\n\n";
  }
  return false;
}

bool emitCompilationTargetOverrides(const llvm::RecordKeeper& records,
                                    llvm::raw_ostream& os) {
  const llvm::Record* targetClass = records.getClass("CompilationTarget");
  if (!targetClass) {
    return false;
  }

  os << kOverridePrelude;

  for (const auto& value : targetClass->getValues()) {
    if (value.isTemplateArg()) {
      continue;
    }
    llvm::StringRef fieldName = value.getName();
    if (fieldName == "backendName") {
      continue;
    }

    const llvm::RecTy* type = value.getType();
    if (llvm::isa<llvm::IntRecTy>(type)) {
      os << llvm::formatv(kIntFieldOverride.data(), fieldName);
    } else if (llvm::isa<llvm::StringRecTy>(type)) {
      os << llvm::formatv(kStringFieldOverride.data(), fieldName);
    } else {
      llvm::errs() << "Warning: unsupported field type for override: "
                   << fieldName << "\n";
      return false;
    }
  }

  os << R"(  return emitError(loc, "Cannot override backend config for unknown key: ") << key;
}
)";

  os << kParserPrelude;

  for (const auto& value : targetClass->getValues()) {
    if (value.isTemplateArg()) {
      continue;
    }
    llvm::StringRef fieldName = value.getName();
    if (fieldName == "backendName") {
      continue;
    }

    const llvm::RecTy* type = value.getType();
    if (llvm::isa<llvm::IntRecTy>(type)) {
      os << llvm::formatv(kIntFieldParser.data(), fieldName);
    } else if (llvm::isa<llvm::StringRecTy>(type)) {
      os << llvm::formatv(kStringFieldParser.data(), fieldName);
    } else {
      llvm::errs() << "Warning: unsupported field type for parser: "
                   << fieldName << "\n";
      return false;
    }
  }

  os << R"(  return emitError(loc, "Cannot parse backend config for unknown key: ") << key;
}

}  // namespace heir
}  // namespace mlir
)";

  return false;
}

bool emitCompilationTargetStruct(const llvm::RecordKeeper& records,
                                 llvm::raw_ostream& os) {
  auto targets = records.getAllDerivedDefinitions("CompilationTarget");

  os << "namespace mlir {\n"
     << "namespace heir {\n\n";

  // Generate BackendName enum
  os << "enum class BackendName {\n"
     << "  None,\n";
  for (auto* target : targets) {
    os << "  " << target->getName() << ",\n";
  }
  os << "};\n\n";

  // Generate toString helper
  os << "inline llvm::StringRef toString(BackendName backend) {\n"
     << "  switch (backend) {\n"
     << "    case BackendName::None: return \"none\";\n";
  for (auto* target : targets) {
    auto name = target->getName();
    auto backendName = target->getValueAsString("backendName");
    os << "    case BackendName::" << name << ": return \"" << backendName
       << "\";\n";
  }
  os << "  }\n"
     << "}\n\n";

  // Generate parseBackendName helper
  os << "inline BackendName parseBackendName(llvm::StringRef name) {\n";
  for (auto* target : targets) {
    auto name = target->getName();
    auto backendName = target->getValueAsString("backendName");
    os << "  if (name == \"" << backendName
       << "\") return BackendName::" << name << ";\n";
  }
  os << "  return BackendName::None;\n"
     << "}\n\n";

  const llvm::Record* targetClass = records.getClass("CompilationTarget");
  if (!targetClass) {
    return false;
  }

  os << "struct CompilationTarget {\n";

  for (const auto& value : targetClass->getValues()) {
    if (value.isTemplateArg()) {
      continue;
    }
    llvm::StringRef fieldName = value.getName();
    const llvm::RecTy* type = value.getType();
    const llvm::Init* init = value.getValue();

    std::string cppType;
    std::string defaultVal = "";

    if (fieldName == "backendName") {
      cppType = "BackendName";
      defaultVal = " = BackendName::None";
    } else if (llvm::isa<llvm::IntRecTy>(type)) {
      cppType = "int64_t";
      if (init && init->isConcrete()) {
        if (auto* intInit = llvm::dyn_cast<llvm::IntInit>(init)) {
          defaultVal = " = " + std::to_string(intInit->getValue());
        }
      }
    } else if (llvm::isa<llvm::StringRecTy>(type)) {
      cppType = "std::string";
      if (init && init->isConcrete()) {
        if (auto* stringInit = llvm::dyn_cast<llvm::StringInit>(init)) {
          defaultVal = " = \"" + stringInit->getValue().str() + "\"";
        }
      }
    } else {
      llvm::errs() << "Warning: unsupported field type for struct: "
                   << fieldName << "\n";
      continue;
    }

    os << "  " << cppType << " " << fieldName << defaultVal << ";\n";
  }

  os << "};\n\n"
     << "}  // namespace heir\n"
     << "}  // namespace mlir\n";

  return false;
}

}  // namespace heir
}  // namespace mlir
