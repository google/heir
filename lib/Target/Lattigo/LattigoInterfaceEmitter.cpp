#include "lib/Target/Lattigo/LattigoInterfaceEmitter.h"

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "lib/Dialect/Lattigo/IR/LattigoTypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Target/Lattigo/LattigoEmitter.h"
#include "lib/Target/Lattigo/LattigoTemplates.h"
#include "lib/Utils/EntryInterfaceUtils.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"           // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/IndentedOstream.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir {
namespace heir {
namespace lattigo {

namespace {

// The context field a support argument is served from; none for an argument
// the caller has to supply.
std::optional<std::string> contextFieldFor(Type type) {
  return llvm::TypeSwitch<Type, std::optional<std::string>>(type)
      .Case<CKKSEvaluatorType, BGVEvaluatorType>(
          [](auto) { return std::string("Evaluator"); })
      .Case<CKKSParameterType, BGVParameterType>(
          [](auto) { return std::string("Params"); })
      .Case<CKKSEncoderType, BGVEncoderType>(
          [](auto) { return std::string("Encoder"); })
      .Case<RLWEEncryptorType>([](auto) { return std::string("Encryptor"); })
      .Case<RLWEDecryptorType>([](auto) { return std::string("Decryptor"); })
      .Case<CKKSBootstrappingEvaluatorType>(
          [](auto) { return std::string("BootEvaluator"); })
      .Default([](Type) { return std::nullopt; });
}

// The elements a data argument carries, as a Go scalar type, or none when the
// argument is not numeric (a ciphertext, say).
FailureOr<std::string> elementGoType(Type type) {
  Type element = getElementTypeOrSelf(type);
  if (!element.isIntOrFloat()) return failure();
  return convertLattigoType(element);
}

struct DataArgument {
  unsigned index;
  Type type;
};

// The arguments of `function` the caller has to provide, in order.
SmallVector<DataArgument> dataArguments(func::FuncOp function) {
  SmallVector<DataArgument> result;
  for (auto [index, type] : llvm::enumerate(function.getArgumentTypes()))
    if (!contextFieldFor(type))
      result.push_back({static_cast<unsigned>(index), type});
  return result;
}

class InterfaceEmitter {
 public:
  InterfaceEmitter(raw_ostream& os, const std::string& packageName,
                   const std::vector<std::string>& extraImports,
                   const std::string& interfacePrefix)
      : finalOs(os),
        bodyOs(body),
        os(bodyOs),
        packageName(packageName),
        extraImports(extraImports),
        prefix(interfacePrefix) {}

  LogicalResult emit(ModuleOp module);

 private:
  raw_ostream& finalOs;
  std::string body;
  llvm::raw_string_ostream bodyOs;
  raw_indented_ostream os;
  const std::string& packageName;
  const std::vector<std::string> extraImports;

  std::string prefix;
  EntryFunctions functions;
  SmallVector<Type> entryArgTypes;
  unsigned numLogicalInputs = 0;
  // (entry argument position, client.enc_zero_arg index) for the arguments
  // holding client-supplied encrypted zeros.
  SmallVector<std::pair<unsigned, unsigned>> zeroArgs;

  std::optional<unsigned> zeroIndexAt(unsigned position) const {
    for (auto [argPosition, zeroIndex] : zeroArgs)
      if (argPosition == position) return zeroIndex;
    return std::nullopt;
  }

  // Whether ctx.<field> is available, i.e. __configure returns it.
  bool hasContextField(StringRef field) const {
    return llvm::is_contained(contextFields, field);
  }
  SmallVector<std::string> contextFields;

  std::string callArgs(func::FuncOp callee, ArrayRef<std::string> data) const;
  // Preprocessing helpers are emitted into a separate package when the
  // preprocessing/preprocessed split is in use.
  std::string calleeName(func::FuncOp callee) const {
    std::string name = toExportName(callee.getSymName());
    if (extraImports.empty() || !isPreprocessingHelper(callee)) return name;
    return packageName + "_utils." + name;
  }
  LogicalResult emitContext();
  LogicalResult emitPrepared();
  LogicalResult emitEncrypted();
  LogicalResult emitEvaluated();
  LogicalResult emitEncrypt();
  LogicalResult emitEncryptedFrom();
  LogicalResult emitPreprocess();
  LogicalResult emitEvaluate();
  LogicalResult emitDecrypt();
  // Emits `name := make([]T, len(src))` plus the conversion loop.
  void emitConversion(StringRef name, StringRef src, StringRef goElement);
};

// Fills in a helper's support arguments from the context and its data
// arguments from `data`, in signature order.
std::string InterfaceEmitter::callArgs(func::FuncOp callee,
                                       ArrayRef<std::string> data) const {
  std::string result;
  unsigned next = 0;
  for (Type type : callee.getArgumentTypes()) {
    if (!result.empty()) result += ", ";
    if (std::optional<std::string> field = contextFieldFor(type)) {
      result += "ctx." + *field;
      continue;
    }
    result += next < data.size() ? data[next] : "nil";
    ++next;
  }
  return result;
}

void InterfaceEmitter::emitConversion(StringRef name, StringRef src,
                                      StringRef goElement) {
  os << name << " := make([]" << goElement << ", len(" << src << "))\n";
  os << "for i, v := range " << src << " {\n";
  os.indent();
  os << name << "[i] = " << goElement << "(v)\n";
  os.unindent();
  os << "}\n";
}

LogicalResult InterfaceEmitter::emitContext() {
  if (!functions.setup)
    return functions.contract.emitOpError(
        "has no client.setup_func; run the configure-crypto-context pass");

  for (Type type : functions.setup.getResultTypes()) {
    std::optional<std::string> field = contextFieldFor(type);
    if (!field)
      return functions.setup.emitOpError()
             << "returns a value the interface has no context field for";
    contextFields.push_back(*field);
  }

  os << "type " << prefix << "Context struct {\n";
  os.indent();
  for (auto [field, type] :
       llvm::zip(contextFields, functions.setup.getResultTypes())) {
    FailureOr<std::string> goType = convertLattigoType(type);
    if (failed(goType)) return failure();
    os << field << " " << *goType << "\n";
  }
  os.unindent();
  os << "}\n\n";

  os << "func " << prefix << "Setup() *" << prefix << "Context {\n";
  os.indent();
  os << "ctx := &" << prefix << "Context{}\n";
  for (auto [i, field] : llvm::enumerate(contextFields))
    os << (i == 0 ? "" : ", ") << "ctx." << field;
  os << " = " << calleeName(functions.setup) << "()\n";
  os << "return ctx\n";
  os.unindent();
  os << "}\n\n";
  return success();
}

LogicalResult InterfaceEmitter::emitPrepared() {
  os << "type " << prefix << "Prepared struct {\n";
  os.indent();
  if (functions.preprocess) {
    for (auto [i, type] :
         llvm::enumerate(functions.preprocess.getResultTypes())) {
      FailureOr<std::string> goType = convertLattigoType(type);
      if (failed(goType)) return failure();
      os << "S" << i << " " << *goType << "\n";
    }
  }
  os.unindent();
  os << "}\n\n";
  return success();
}

LogicalResult InterfaceEmitter::emitEncrypted() {
  os << "type " << prefix << "Encrypted struct {\n";
  os.indent();
  for (unsigned i = 0; i < entryArgTypes.size(); ++i) {
    if (zeroIndexAt(i)) continue;
    func::FuncOp helper = findIndexedHelper(functions.inputHelpers, i);
    Type type = helper ? helper.getResultTypes().front() : entryArgTypes[i];
    FailureOr<std::string> goType = convertLattigoType(type);
    if (failed(goType)) return failure();
    os << "Arg" << i << " " << *goType << "\n";
  }
  for (auto [position, zeroIndex] : zeroArgs) {
    FailureOr<std::string> goType = convertLattigoType(entryArgTypes[position]);
    if (failed(goType)) return failure();
    os << "Zero" << zeroIndex << " " << *goType << "\n";
  }
  os.unindent();
  os << "}\n\n";
  return success();
}

LogicalResult InterfaceEmitter::emitEvaluated() {
  os << "type " << prefix << "Evaluated struct {\n";
  os.indent();
  for (auto [i, type] : llvm::enumerate(functions.evaluate.getResultTypes())) {
    FailureOr<std::string> goType = convertLattigoType(type);
    if (failed(goType)) return failure();
    os << "Res" << i << " " << *goType << "\n";
  }
  os.unindent();
  os << "}\n\n";
  return success();
}

LogicalResult InterfaceEmitter::emitEncrypt() {
  if (functions.inputHelpers.empty()) return success();
  // Every logical input needs either an encryption helper or a numeric type we
  // can pass through; a program entered at ciphertext semantics has neither.
  for (unsigned i = 0; i < numLogicalInputs; ++i) {
    if (findIndexedHelper(functions.inputHelpers, i)) continue;
    if (failed(elementGoType(entryArgTypes[i]))) return success();
  }
  if (!hasContextField("Encryptor")) return success();

  os << "func (ctx *" << prefix << "Context) Encrypt(inputs [][]float64) "
     << prefix << "Encrypted {\n";
  os.indent();
  os << "if len(inputs) != " << numLogicalInputs << " {\n";
  os.indent();
  os << "panic(fmt.Sprintf(\"" << functions.entryName << ": expected "
     << numLogicalInputs << " inputs, got %d\", len(inputs)))\n";
  os.unindent();
  os << "}\n";
  os << "var enc " << prefix << "Encrypted\n";
  for (unsigned i = 0; i < numLogicalInputs; ++i) {
    func::FuncOp helper = findIndexedHelper(functions.inputHelpers, i);
    Type target = helper ? helper.getArgumentTypes().back() : entryArgTypes[i];
    FailureOr<std::string> element = elementGoType(target);
    if (failed(element)) return failure();
    std::string local = "arg" + std::to_string(i);
    emitConversion(local, "inputs[" + std::to_string(i) + "]", *element);
    if (helper) {
      os << "enc.Arg" << i << " = " << calleeName(helper) << "("
         << callArgs(helper, {local}) << ")\n";
    } else {
      os << "enc.Arg" << i << " = " << local << "\n";
    }
  }
  for (auto [position, zeroIndex] : zeroArgs) {
    func::FuncOp helper = findIndexedHelper(functions.zeroHelpers, zeroIndex);
    if (!helper)
      return functions.contract.emitOpError()
             << "has no encrypted-zero helper with index " << zeroIndex;
    os << "enc.Zero" << zeroIndex << " = " << calleeName(helper) << "("
       << callArgs(helper, {}) << ")\n";
  }
  os << "return enc\n";
  os.unindent();
  os << "}\n\n";
  return success();
}

// A program entered at ciphertext semantics has no encryption helpers, so its
// harness encrypts for itself and hands the ciphertexts back here.
LogicalResult InterfaceEmitter::emitEncryptedFrom() {
  for (unsigned i = 0; i < entryArgTypes.size(); ++i)
    if (!zeroIndexAt(i) && !isa<RLWECiphertextType>(entryArgTypes[i]) &&
        failed(elementGoType(entryArgTypes[i])))
      return success();

  os << "func (ctx *" << prefix
     << "Context) EncryptedFrom(cts []*rlwe.Ciphertext, args [][]float64) "
     << prefix << "Encrypted {\n";
  os.indent();
  os << "var enc " << prefix << "Encrypted\n";
  unsigned nextCiphertext = 0;
  for (unsigned i = 0; i < entryArgTypes.size(); ++i) {
    if (zeroIndexAt(i)) continue;
    if (isa<RLWECiphertextType>(entryArgTypes[i])) {
      os << "enc.Arg" << i << " = cts[" << nextCiphertext++ << "]\n";
      continue;
    }
    std::string local = "arg" + std::to_string(i);
    emitConversion(local, "args[" + std::to_string(i) + "]",
                   *elementGoType(entryArgTypes[i]));
    os << "enc.Arg" << i << " = " << local << "\n";
  }
  for (auto [position, zeroIndex] : zeroArgs) {
    func::FuncOp helper = findIndexedHelper(functions.zeroHelpers, zeroIndex);
    if (!helper)
      return functions.contract.emitOpError()
             << "has no encrypted-zero helper with index " << zeroIndex;
    os << "enc.Zero" << zeroIndex << " = " << calleeName(helper) << "("
       << callArgs(helper, {}) << ")\n";
  }
  os << "return enc\n";
  os.unindent();
  os << "}\n\n";
  return success();
}

LogicalResult InterfaceEmitter::emitPreprocess() {
  // Emitted even without a preprocessing function, so that a harness calls the
  // same sequence whether or not the program was split.
  if (!functions.preprocess) {
    os << "func (ctx *" << prefix << "Context) Preprocess(inputs [][]float64) "
       << prefix << "Prepared {\n";
    os.indent();
    os << "return " << prefix << "Prepared{}\n";
    os.unindent();
    os << "}\n\n";
    return success();
  }

  DictionaryAttr role =
      getRoleAttr(functions.preprocess, kServerPreprocessingFuncAttrName);
  auto entryArgs = dyn_cast_or_null<DenseI64ArrayAttr>(
      role.get(kServerPreprocessingEntryArgs));
  SmallVector<DataArgument> data = dataArguments(functions.preprocess);
  if (!entryArgs || entryArgs.size() != static_cast<int64_t>(data.size()))
    return functions.preprocess.emitOpError()
           << "server.preprocessing_func is missing entry_arg_indices";

  os << "func (ctx *" << prefix << "Context) Preprocess(inputs [][]float64) "
     << prefix << "Prepared {\n";
  os.indent();
  SmallVector<std::string> arguments;
  for (auto [i, argument] : llvm::enumerate(data)) {
    int64_t entryIndex = entryArgs[i];
    FailureOr<std::string> element = elementGoType(argument.type);
    if (entryIndex < 0 || failed(element))
      return functions.preprocess.emitOpError()
             << "parameter " << i << " is not forwarded from an entry argument";
    std::string local = "arg" + std::to_string(entryIndex);
    emitConversion(local, "inputs[" + std::to_string(entryIndex) + "]",
                   *element);
    arguments.push_back(local);
  }
  os << "var prep " << prefix << "Prepared\n";
  for (unsigned i = 0; i < functions.preprocess.getNumResults(); ++i)
    os << (i == 0 ? "" : ", ") << "prep.S" << i;
  os << " = " << calleeName(functions.preprocess) << "("
     << callArgs(functions.preprocess, arguments) << ")\n";
  os << "return prep\n";
  os.unindent();
  os << "}\n\n";
  return success();
}

LogicalResult InterfaceEmitter::emitEvaluate() {
  SmallVector<DataArgument> data = dataArguments(functions.evaluate);
  // The evaluate function takes the entry arguments, then the preprocessing
  // storage the split appended.
  unsigned numStorage =
      functions.preprocess ? functions.preprocess.getNumResults() : 0;
  if (data.size() != entryArgTypes.size() + numStorage)
    return functions.evaluate.emitOpError()
           << "takes " << data.size() << " data arguments, expected "
           << entryArgTypes.size() + numStorage;

  os << "func (ctx *" << prefix << "Context) Evaluate(prep " << prefix
     << "Prepared, enc " << prefix << "Encrypted) " << prefix
     << "Evaluated {\n";
  os.indent();
  SmallVector<std::string> arguments;
  for (unsigned i = 0; i < entryArgTypes.size(); ++i) {
    std::optional<unsigned> zeroIndex = zeroIndexAt(i);
    arguments.push_back(zeroIndex ? "enc.Zero" + std::to_string(*zeroIndex)
                                  : "enc.Arg" + std::to_string(i));
  }
  for (unsigned i = 0; i < numStorage; ++i)
    arguments.push_back("prep.S" + std::to_string(i));

  os << "var out " << prefix << "Evaluated\n";
  for (unsigned i = 0; i < functions.evaluate.getNumResults(); ++i)
    os << (i == 0 ? "" : ", ") << "out.Res" << i;
  os << " = " << calleeName(functions.evaluate) << "("
     << callArgs(functions.evaluate, arguments) << ")\n";
  os << "return out\n";
  os.unindent();
  os << "}\n\n";
  return success();
}

LogicalResult InterfaceEmitter::emitDecrypt() {
  if (functions.outputHelpers.size() != functions.evaluate.getNumResults())
    return success();
  if (!hasContextField("Decryptor")) return success();

  os << "func (ctx *" << prefix << "Context) Decrypt(out " << prefix
     << "Evaluated) [][]float64 {\n";
  os.indent();
  os << "results := make([][]float64, " << functions.outputHelpers.size()
     << ")\n";
  for (auto [index, helper] : functions.outputHelpers) {
    FailureOr<std::string> element = elementGoType(helper.getResultTypes()[0]);
    if (failed(element)) return failure();
    std::string local = "res" + std::to_string(index);
    os << local << " := " << calleeName(helper) << "("
       << callArgs(helper, {"out.Res" + std::to_string(index)}) << ")\n";
    os << "results[" << index << "] = make([]float64, len(" << local << "))\n";
    os << "for i, v := range " << local << " {\n";
    os.indent();
    os << "results[" << index << "][i] = float64(v)\n";
    os.unindent();
    os << "}\n";
  }
  os << "return results\n";
  os.unindent();
  os << "}\n\n";
  return success();
}

LogicalResult InterfaceEmitter::emit(ModuleOp module) {
  FailureOr<EntryFunctions> found = findEntryFunctions(module, "");
  if (failed(found)) return failure();
  functions = *found;
  if (prefix.empty()) prefix = toExportName(functions.entryName);

  if (functions.contract) {
    ArrayAttr inputTypes =
        getLogicalTypes(functions.contract, kEntryInputTypesAttrName);
    if (!inputTypes)
      return functions.contract.emitOpError(
          "is missing heir.entry_input_types");
    numLogicalInputs = inputTypes.size();

    for (auto [index, type] :
         llvm::enumerate(functions.contract.getArgumentTypes())) {
      if (contextFieldFor(type)) continue;
      if (auto role = functions.contract.getArgAttrOfType<DictionaryAttr>(
              index, kClientEncZeroArgAttrName)) {
        auto zeroIndex =
            dyn_cast_or_null<IntegerAttr>(role.get(kClientHelperIndex));
        if (!zeroIndex)
          return functions.contract.emitOpError()
                 << "argument " << index << " has no encrypted-zero index";
        zeroArgs.push_back({static_cast<unsigned>(entryArgTypes.size()),
                            static_cast<unsigned>(zeroIndex.getInt())});
      }
      entryArgTypes.push_back(type);
    }
    if (entryArgTypes.size() < numLogicalInputs)
      return functions.contract.emitOpError()
             << "has fewer arguments than heir.entry_input_types";
    if (failed(validateIndexedHelpers(functions.inputHelpers, numLogicalInputs,
                                      "encryption", functions.contract)) ||
        failed(validateIndexedHelpers(functions.outputHelpers,
                                      functions.contract.getNumResults(),
                                      "decryption", functions.contract)))
      return failure();
  } else {
    // Without a contract the evaluate function is the only record of the
    // entry's arguments: it takes them all, followed by the preprocessing
    // storage the split appended.
    unsigned numStorage =
        functions.preprocess ? functions.preprocess.getNumResults() : 0;
    SmallVector<DataArgument> data = dataArguments(functions.evaluate);
    if (data.size() < numStorage)
      return functions.evaluate.emitOpError()
             << "takes fewer arguments than the preprocessing storage";
    for (const DataArgument& argument :
         ArrayRef<DataArgument>(data).drop_back(numStorage))
      entryArgTypes.push_back(argument.type);
  }

  if (failed(emitContext()) || failed(emitPrepared()) ||
      failed(emitEncrypted()) || failed(emitEvaluated()) ||
      failed(emitEncrypt()) || failed(emitEncryptedFrom()) ||
      failed(emitPreprocess()) || failed(emitEvaluate()) ||
      failed(emitDecrypt()))
    return failure();

  finalOs << "package " << packageName << "\n\n";
  finalOs << "import (\n";
  // Go rejects unused imports, so take them from what the body mentions.
  for (std::string_view import :
       {std::string_view("\"fmt\""), kRlweImport, kCkksImport, kBgvImport,
        kLintransImport, kBootstrappingImport}) {
    StringRef path = StringRef(import).trim('"');
    StringRef qualifier = path.contains('/') ? path.rsplit('/').second : path;
    if (StringRef(body).contains(qualifier.str() + "."))
      finalOs << "\t" << import << "\n";
  }
  for (const std::string& extraImport : extraImports)
    finalOs << "\t\"" << extraImport << "\"\n";
  finalOs << ")\n\n";
  finalOs << body;
  return success();
}

}  // namespace

LogicalResult translateToLattigoInterface(
    Operation* op, llvm::raw_ostream& os, const std::string& packageName,
    const std::vector<std::string>& extraImports,
    const std::string& interfacePrefix) {
  auto module = dyn_cast<ModuleOp>(op);
  if (!module) return op->emitError("expected a module");
  InterfaceEmitter emitter(os, packageName, extraImports, interfacePrefix);
  return emitter.emit(module);
}

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir
