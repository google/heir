#include "lib/Target/OpenFheEmitC/OpenFhePybindEmitter.h"

#include <map>
#include <set>
#include <string>
#include <vector>

#include "mlir/include/mlir/Dialect/EmitC/IR/EmitC.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Target/Cpp/CppEmitter.h"    // from @llvm-project

namespace mlir::heir::openfhe {

namespace {

std::string stripStruct(const std::string& typeStr) {
  if (typeStr.rfind("struct ", 0) == 0) {
    return typeStr.substr(7);
  }
  return typeStr;
}

void collectTypes(Type type, std::set<std::string>& opaqueTypes,
                  std::set<std::string>& structTypes) {
  if (auto ptrType = dyn_cast<emitc::PointerType>(type)) {
    collectTypes(ptrType.getPointee(), opaqueTypes, structTypes);
    return;
  }
  if (auto opaqueType = dyn_cast<emitc::OpaqueType>(type)) {
    std::string typeStr = opaqueType.getValue().str();
    if (typeStr.rfind("struct ", 0) == 0) {
      structTypes.insert(typeStr);
    } else if (typeStr.rfind("std::", 0) == 0) {
      // skip
    } else {
      opaqueTypes.insert(typeStr);
    }
  }
}

std::string cppTypeName(Type type) {
  if (auto opaqueType = dyn_cast<emitc::OpaqueType>(type)) {
    return opaqueType.getValue().str();
  }
  if (auto ptrType = dyn_cast<emitc::PointerType>(type)) {
    return cppTypeName(ptrType.getPointee()) + "*";
  }
  if (type.isF32()) return "float";
  if (type.isF64()) return "double";
  if (type.isInteger(32)) return "int32_t";
  if (type.isInteger(64)) return "int64_t";
  if (type.isIndex()) return "size_t";

  std::string typeStr;
  llvm::raw_string_ostream os(typeStr);
  type.print(os);
  return typeStr;
}

void emitCommonBindings(llvm::raw_ostream& os) {
  os << R"cpp(
    using namespace lbcrypto;

    // Minimal bindings required for generated functions to run.
    void bind_common(py::module& m) {
      py::class_<PublicKeyImpl<DCRTPoly>,
                 std::shared_ptr<PublicKeyImpl<DCRTPoly>>>(m, "PublicKey",
                                                           py::module_local())
          .def(py::init<>());
      py::class_<PrivateKeyImpl<DCRTPoly>,
                 std::shared_ptr<PrivateKeyImpl<DCRTPoly>>>(m, "PrivateKey",
                                                            py::module_local())
          .def(py::init<>());
      py::class_<KeyPair<DCRTPoly>>(m, "KeyPair", py::module_local())
          .def_readwrite("publicKey", &KeyPair<DCRTPoly>::publicKey)
          .def_readwrite("secretKey", &KeyPair<DCRTPoly>::secretKey);
      py::class_<CiphertextImpl<DCRTPoly>,
                 std::shared_ptr<CiphertextImpl<DCRTPoly>>>(m, "Ciphertext",
                                                            py::module_local())
          .def(py::init<>());
      py::class_<PlaintextImpl, std::shared_ptr<PlaintextImpl>>(
          m, "Plaintext", py::module_local());
      py::class_<CryptoContextImpl<DCRTPoly>,
                 std::shared_ptr<CryptoContextImpl<DCRTPoly>>>(
          m, "CryptoContext", py::module_local())
          .def(py::init<>())
          .def("KeyGen", &CryptoContextImpl<DCRTPoly>::KeyGen);
    }
  )cpp";
}

bool isFundamentalType(Type type) {
  return type.isF32() || type.isF64() || type.isInteger(32) ||
         type.isInteger(64) || type.isIndex();
}

bool isSmartPointerType(Type type) {
  if (auto opaqueType = dyn_cast<emitc::OpaqueType>(type)) {
    std::string typeStr = opaqueType.getValue().str();
    return typeStr == "CiphertextT" || typeStr == "Plaintext" ||
           typeStr == "PlaintextT" || typeStr == "PublicKeyT" ||
           typeStr == "PrivateKeyT";
  }
  return false;
}

}  // namespace

LogicalResult translateToOpenFhePybind(Operation* op, llvm::raw_ostream& os) {
  auto topModule = dyn_cast<ModuleOp>(op);
  if (!topModule) {
    return op->emitOpError("expected a top-level ModuleOp");
  }

  // Find the nested pybind module
  ModuleOp pybindModule = nullptr;
  for (auto nestedModule : topModule.getOps<ModuleOp>()) {
    if (nestedModule->hasAttr("heir.pybind_module")) {
      pybindModule = nestedModule;
      break;
    }
  }

  if (!pybindModule) {
    return topModule->emitError(
        "could not find nested module with 'heir.pybind_module' attribute");
  }

  // Extract metadata
  auto moduleNameAttr =
      pybindModule->getAttrOfType<StringAttr>("pybind.module_name");
  if (!moduleNameAttr) {
    return pybindModule->emitError("expected 'pybind.module_name' attribute");
  }
  std::string moduleName = moduleNameAttr.getValue().str();

  auto importsAttr = pybindModule->getAttrOfType<ArrayAttr>("pybind.imports");

  // Collect types and functions from the nested module
  std::set<std::string> opaqueTypes;
  std::set<std::string> structTypes;

  struct FuncInfo {
    std::string name;
    Type returnType;
    std::vector<Type> argTypes;
    std::vector<int64_t> returnShape;
  };
  std::vector<FuncInfo> functions;

  pybindModule.walk([&](emitc::FuncOp funcOp) {
    std::vector<Type> argTypes;
    for (Type argType : funcOp.getArgumentTypes()) {
      argTypes.push_back(argType);
      collectTypes(argType, opaqueTypes, structTypes);
    }
    Type returnType = nullptr;
    if (funcOp.getNumResults() > 0) {
      returnType = funcOp.getResultTypes()[0];
      collectTypes(returnType, opaqueTypes, structTypes);
    }
    std::vector<int64_t> returnShape;
    if (auto shapeAttr =
            funcOp->getAttrOfType<ArrayAttr>("pybind.return_shape")) {
      for (auto attr : shapeAttr) {
        returnShape.push_back(cast<IntegerAttr>(attr).getInt());
      }
    }
    functions.push_back(
        {funcOp.getName().str(), returnType, argTypes, returnShape});
  });

  // Collect class definitions from the top-level module to resolve struct
  // fields
  std::map<std::string, emitc::ClassOp> classMap;
  topModule.walk([&classMap](emitc::ClassOp classOp) {
    classMap[classOp.getName().str()] = classOp;
  });

  // Emit headers
  os << "#include <pybind11/pybind11.h>\n";
  os << "#include <pybind11/stl.h>\n";
  os << "#include <utility>\n";
  os << "#include <cstdlib>\n";
  if (importsAttr) {
    for (auto importAttr : importsAttr.getAsRange<StringAttr>()) {
      os << "#include \"" << importAttr.getValue() << "\"\n";
    }
  }
  os << "\n";
  os << "namespace py = pybind11;\n\n";

  // Emit forward declarations and structs from the nested module
  if (failed(mlir::emitc::translateToCpp(pybindModule, os))) {
    return pybindModule->emitError("failed to translate nested module to C++");
  }
  os << "\n";

  // Emit common OpenFHE bindings
  emitCommonBindings(os);
  os << "\n";

  // Emit module definition
  os << "PYBIND11_MODULE(" << moduleName << ", m) {\n";
  os << "  bind_common(m);\n";

  // Emit opaque type registrations
  for (const auto& opaqueType : opaqueTypes) {
    if (opaqueType == "CryptoContextT" || opaqueType == "CiphertextT" ||
        opaqueType == "Plaintext" || opaqueType == "PlaintextT" ||
        opaqueType == "PublicKeyT" || opaqueType == "PrivateKeyT") {
      continue;
    }
    os << "  py::class_<" << opaqueType << ">(m, \"" << opaqueType << "\");\n";
  }

  // Emit struct registrations
  for (const auto& structType : structTypes) {
    std::string cleanName = stripStruct(structType);
    os << "  py::class_<" << cleanName << ">(m, \"" << cleanName
       << "\", py::module_local())\n";

    // Find fields
    auto it = classMap.find(cleanName);
    if (it != classMap.end()) {
      for (auto fieldOp : it->second.getOps<emitc::FieldOp>()) {
        std::string fieldName = fieldOp.getName().str();
        Type fieldType = fieldOp.getType();
        bool wrapField = false;
        if (auto ptrType = dyn_cast<emitc::PointerType>(fieldType)) {
          if (auto opaqueType =
                  dyn_cast<emitc::OpaqueType>(ptrType.getPointee())) {
            std::string typeStr = opaqueType.getValue().str();
            if (typeStr == "CiphertextT" || typeStr == "Plaintext" ||
                typeStr == "PlaintextT" || typeStr == "PublicKeyT" ||
                typeStr == "PrivateKeyT") {
              wrapField = true;
            }
          }
        }

        if (wrapField) {
          auto ptrType = cast<emitc::PointerType>(fieldType);
          std::string pointeeType = cppTypeName(ptrType.getPointee());
          os << "    .def_property(\"" << fieldName << "\",\n"
             << "      [](const " << cleanName << "& s) { return s."
             << fieldName << " ? *s." << fieldName << " : " << pointeeType
             << "(); },\n"
             << "      []( " << cleanName << "& s, const " << pointeeType
             << "& val) { if (s." << fieldName << ") *s." << fieldName
             << " = val; })\n";
        } else {
          os << "    .def_readwrite(\"" << fieldName << "\", &" << cleanName
             << "::" << fieldName << ")\n";
        }
      }
    }
    os << "    ;\n";  // end py::class_
  }

  // Emit function bindings
  for (const auto& func : functions) {
    bool wrapInLambda = false;
    std::string pointeeTypeName = "";

    // 1. Detect if we need to wrap due to return type (pointer to smart
    // pointer)
    if (func.returnType) {
      if (auto ptrType = dyn_cast<emitc::PointerType>(func.returnType)) {
        if (auto opaqueType =
                dyn_cast<emitc::OpaqueType>(ptrType.getPointee())) {
          std::string typeStr = opaqueType.getValue().str();
          if (typeStr == "CiphertextT" || typeStr == "Plaintext" ||
              typeStr == "PlaintextT" || typeStr == "PublicKeyT" ||
              typeStr == "PrivateKeyT") {
            wrapInLambda = true;
            pointeeTypeName = typeStr;
          }
        }
      }
    }

    // 2. Detect if we need to wrap due to returning a shaped type (fundamental
    // pointer with shape)
    bool returnFundamentalPtrWithShape = false;
    if (func.returnType && !func.returnShape.empty()) {
      if (auto ptrType = dyn_cast<emitc::PointerType>(func.returnType)) {
        if (isFundamentalType(ptrType.getPointee())) {
          wrapInLambda = true;
          returnFundamentalPtrWithShape = true;
        }
      }
    }

    // 3. Detect if we need to wrap due to arguments (pointer to fundamental
    // type or pointer to smart pointer)
    for (Type argType : func.argTypes) {
      if (auto ptrType = dyn_cast<emitc::PointerType>(argType)) {
        if (isFundamentalType(ptrType.getPointee()) ||
            isSmartPointerType(ptrType.getPointee())) {
          wrapInLambda = true;
        }
      }
    }

    if (wrapInLambda) {
      os << "  m.def(\"" << func.name << "\", [](";
      for (size_t i = 0; i < func.argTypes.size(); ++i) {
        if (i > 0) os << ", ";
        Type argType = func.argTypes[i];
        if (auto ptrType = dyn_cast<emitc::PointerType>(argType)) {
          if (isFundamentalType(ptrType.getPointee())) {
            os << "std::vector<" << cppTypeName(ptrType.getPointee()) << "> arg"
               << i;
            continue;
          }
          if (isSmartPointerType(ptrType.getPointee())) {
            os << cppTypeName(ptrType.getPointee()) << " arg" << i;
            continue;
          }
        }
        os << cppTypeName(argType) << " arg" << i;
      }

      // Determine the C++ return type of the lambda
      std::string lambdaReturnType = "";
      if (func.returnType) {
        if (!pointeeTypeName.empty()) {
          lambdaReturnType = pointeeTypeName;
        } else if (returnFundamentalPtrWithShape) {
          auto ptrType = cast<emitc::PointerType>(func.returnType);
          lambdaReturnType =
              "std::vector<" + cppTypeName(ptrType.getPointee()) + ">";
        } else {
          lambdaReturnType = cppTypeName(func.returnType);
        }
      }

      if (lambdaReturnType.empty()) {
        os << ") {\n";
      } else {
        os << ") -> " << lambdaReturnType << " {\n";
      }

      if (func.returnType) {
        if (!pointeeTypeName.empty() || returnFundamentalPtrWithShape) {
          os << "    auto* res_ptr = ";
        } else {
          os << "    auto res = ";
        }
      }
      os << func.name << "(";
      for (size_t i = 0; i < func.argTypes.size(); ++i) {
        if (i > 0) os << ", ";
        Type argType = func.argTypes[i];
        if (auto ptrType = dyn_cast<emitc::PointerType>(argType)) {
          if (isFundamentalType(ptrType.getPointee())) {
            os << "arg" << i << ".data()";
            continue;
          }
          if (isSmartPointerType(ptrType.getPointee())) {
            os << "&arg" << i;
            continue;
          }
        }
        os << "arg" << i;
      }
      os << ");\n";

      if (func.returnType) {
        if (!pointeeTypeName.empty()) {
          os << "    auto res = *res_ptr;\n";
          os << "    res_ptr->~" << pointeeTypeName << "();\n";
          os << "    free(res_ptr);\n";
          os << "    return res;\n";
        } else if (returnFundamentalPtrWithShape) {
          // Compute total size of the shape
          int64_t totalSize = 1;
          for (auto dim : func.returnShape) {
            totalSize *= dim;
          }
          auto ptrType = cast<emitc::PointerType>(func.returnType);
          std::string eltType = cppTypeName(ptrType.getPointee());
          os << "    std::vector<" << eltType << "> res(res_ptr, res_ptr + "
             << totalSize << ");\n";
          os << "    free(res_ptr);\n";
          os << "    return res;\n";
        } else {
          os << "    return res;\n";
        }
      }
      os << "  }, py::call_guard<py::gil_scoped_release>());\n";
    } else {
      os << "  m.def(\"" << func.name << "\", &" << func.name
         << ", py::call_guard<py::gil_scoped_release>());\n";
    }
  }

  os << "}\n";

  return success();
}

}  // namespace mlir::heir::openfhe
