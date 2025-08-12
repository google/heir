#include "lib/Target/Metadata/MetadataEmitter.h"

#include <utility>

#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"  // from @llvm-project
#include "llvm/include/llvm/Support/JSON.h"            // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project

namespace mlir {
namespace heir {

namespace {

using ::mlir::ModuleOp;
using ::mlir::func::FuncOp;

///
/// The original inspiration for this came from an xlscc protocol buffer
/// of the form
///
/// {
///   top_func_proto: {
///     name: {
///       name    : "real_main"
///     }
///     return_type: {
///       as_int: {
///         width    : 8
///         is_signed: true
///       }
///     }
///     params: {
///       name: "arg1"
///       type: {
///         as_int: {
///           width    : 8
///           is_signed: true
///         }
///       }
///       is_reference: false
///       is_const    : false
///     }
///   }
/// }
///
/// The json structure here is mirroring the relevant parts of that structure,
/// with the exception that MLIR-absent constructions (like function argument
/// names and reference/const) are removed.
///
/// {
///   "functions": [
///     {
///       "name": "real_main",
///       "return_type": {
///         "integer": {
///           "width": 8,
///           "is_signed": true
///         }
///       },
///       "params": [
///         {
///           "type": {
///             "integer": {
///               "width": 8,
///               "is_signed": true
///             }
///           },
///         },
///         ...
///       ]
///     },
///     ...
///   ]
/// }
///

}  // namespace

void registerMetadataEmitter() {
  mlir::TranslateFromMLIRRegistration reg(
      "emit-metadata", "emit function signature metadata for the given MLIR",
      [](Operation* op, llvm::raw_ostream& output) {
        return emitMetadata(op, output);
      },
      [](mlir::DialectRegistry& registry) {
        registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                        mlir::memref::MemRefDialect,
                        mlir::affine::AffineDialect, mlir::scf::SCFDialect>();
      });
}

LogicalResult emitMetadata(Operation* op, llvm::raw_ostream& os) {
  MetadataEmitter emitter;
  FailureOr<llvm::json::Object> result = emitter.translate(*op);
  if (failed(result)) {
    return failure();
  }
  llvm::json::Value output = std::move(result.value());
  os << llvm::formatv("{0:2}", output);
  return success();
}

FailureOr<llvm::json::Object> MetadataEmitter::translate(Operation& op) {
  return llvm::TypeSwitch<Operation&, FailureOr<llvm::json::Object>>(op)
      .Case<ModuleOp, FuncOp>([&](auto op) { return emitOperation(op); })
      .Default([&](Operation&) {
        // Empty object is a sentinel for "skip"
        return llvm::json::Object{};
      });
}

FailureOr<llvm::json::Object> MetadataEmitter::emitOperation(
    ModuleOp moduleOp) {
  llvm::json::Array functions;
  for (Operation& op : moduleOp) {
    auto result = translate(op);
    if (failed(result)) {
      return failure();
    }

    // The op is unsupported
    if (result.value().empty()) {
      continue;
    }

    functions.push_back(std::move(result.value()));
  }

  llvm::json::Object metadata{{"functions", std::move(functions)}};
  return std::move(metadata);
}

FailureOr<llvm::json::Object> MetadataEmitter::typeAsJson(IntegerType& ty) {
  llvm::json::Object output{
      {"integer", llvm::json::Object{{"width", ty.getWidth()},
                                     {"is_signed", ty.isSigned()}}},
  };
  return std::move(output);
}

FailureOr<llvm::json::Object> MetadataEmitter::typeAsJson(MemRefType& ty) {
  auto elementType =
      llvm::TypeSwitch<Type, FailureOr<llvm::json::Object>>(ty.getElementType())
          .Case<IntegerType, MemRefType>([&](auto t) { return typeAsJson(t); })
          .Default([&](Type) { return failure(); });

  if (failed(elementType)) {
    return failure();
  }

  llvm::json::Object output{
      {"memref",
       llvm::json::Object{{"shape", llvm::json::Array{ty.getShape()}},
                          {"element_type", std::move(elementType.value())}}},
  };
  return std::move(output);
}

FailureOr<llvm::json::Object> MetadataEmitter::emitOperation(FuncOp funcOp) {
  llvm::json::Array arguments;
  for (auto arg : funcOp.getArguments()) {
    Type type = arg.getType();
    auto result = llvm::TypeSwitch<Type&, FailureOr<llvm::json::Object>>(type)
                      .Case<IntegerType, MemRefType>(
                          [&](auto ty) { return typeAsJson(ty); })
                      .Default([&](Type&) { return failure(); });

    if (failed(result)) {
      funcOp.emitOpError(
          llvm::formatv("Failed to handle argument type {0}", type));
      return failure();
    }
    arguments.push_back(llvm::json::Object{
        {"index", arg.getArgNumber()},
        {"type", std::move(result.value())},
    });
  }

  auto resultTypes = funcOp.getFunctionType().getResults();
  llvm::json::Array resultsJson;
  if (resultTypes.size() > 1) {
    emitError(funcOp.getLoc(),
              "Only functions with <=1 return types are supported");
    return failure();
  }

  if (resultTypes.size() == 1) {
    auto outputType = resultTypes[0];
    auto status =
        llvm::TypeSwitch<Type&, FailureOr<llvm::json::Object>>(outputType)
            .Case<IntegerType, MemRefType>(
                [&](auto ty) { return typeAsJson(ty); })
            .Default([&](Type&) { return failure(); });

    if (failed(status)) {
      funcOp.emitOpError(
          llvm::formatv("Failed to handle output type {0}", outputType));
      return failure();
    }
    resultsJson.push_back(std::move(status.value()));
  }

  return llvm::json::Object{
      {"name", funcOp.getName()},
      {"return_types", std::move(resultsJson)},
      {"params", std::move(arguments)},
  };
}

}  // namespace heir
}  // namespace mlir
