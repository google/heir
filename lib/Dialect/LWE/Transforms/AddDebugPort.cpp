#include "lib/Dialect/LWE/Transforms/AddDebugPort.h"

#include <string>

#include "lib/Dialect/Debug/IR/DebugOps.h"
#include "lib/Dialect/FuncUtils.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "llvm/include/llvm/ADT/DenseMap.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"            // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"        // from @llvm-project

namespace mlir {
namespace heir {
namespace lwe {

#define GEN_PASS_DEF_ADDDEBUGPORT
#include "lib/Dialect/LWE/Transforms/Passes.h.inc"

FailureOr<Type> getPrivateKeyType(func::FuncOp op) {
  const auto* type = llvm::find_if(op.getArgumentTypes(), [](Type type) {
    return isa<LWECiphertextType>(getElementTypeOrSelf(type));
  });

  if (type == op.getArgumentTypes().end()) {
    return op.emitError(
        "Function does not have an argument of LWECiphertextType");
  }

  auto lweCiphertextType = cast<LWECiphertextType>(getElementTypeOrSelf(*type));

  auto lwePrivateKeyType =
      LWESecretKeyType::get(op.getContext(), lweCiphertextType.getKey(),
                            lweCiphertextType.getCiphertextSpace().getRing());
  return lwePrivateKeyType;
}

func::FuncOp getOrCreateExternalDebugFunc(
    ModuleOp module, Type lwePrivateKeyType, Type valueType,
    llvm::DenseMap<std::pair<Type, Type>, int>& typePairToInt) {
  auto key = std::make_pair(lwePrivateKeyType, valueType);
  if (typePairToInt.find(key) == typePairToInt.end()) {
    typePairToInt[key] = typePairToInt.size();
  }

  std::string funcName = "__heir_debug_" + std::to_string(typePairToInt[key]);

  auto* context = module.getContext();
  auto lookup = module.lookupSymbol<func::FuncOp>(funcName);
  if (lookup) return lookup;

  auto debugFuncType =
      FunctionType::get(context, {lwePrivateKeyType, valueType}, {});

  ImplicitLocOpBuilder b =
      ImplicitLocOpBuilder::atBlockBegin(module.getLoc(), module.getBody());
  auto funcOp = func::FuncOp::create(b, funcName, debugFuncType);
  // required for external func call
  funcOp.setPrivate();
  return funcOp;
}

void insertValidationOps(func::FuncOp op) {
  int count = 0;
  auto insertValidate = [&](Value value, OpBuilder& b) {
    Type valueType = value.getType();
    if (isa<LWECiphertextType>(getElementTypeOrSelf(valueType))) {
      debug::ValidateOp::create(b, value.getLoc(), value,
                                "heir_debug_" + std::to_string(count++),
                                nullptr);
    }
  };

  Block& entryBlock = op.getBody().getBlocks().front();
  OpBuilder argBuilder(&entryBlock, entryBlock.begin());
  for (auto arg : op.getArguments()) {
    insertValidate(arg, argBuilder);
  }

  op.walk([&](Operation* walkOp) {
    if (walkOp == op.getOperation() || isa<tensor::EmptyOp>(walkOp) ||
        walkOp->hasTrait<OpTrait::IsTerminator>())
      return;
    OpBuilder opBuilder(walkOp->getBlock(), ++walkOp->getIterator());
    for (Value result : walkOp->getResults()) {
      insertValidate(result, opBuilder);
    }
  });
}

LogicalResult lowerValidationOps(
    func::FuncOp op, Value privateKey, int messageSize,
    llvm::DenseMap<std::pair<Type, Type>, int>& typePairToInt) {
  auto module = op->getParentOfType<ModuleOp>();
  Type lwePrivateKeyType = privateKey.getType();

  auto walkResult = op.walk([&](debug::ValidateOp validateOp) {
    Value value = validateOp.getInput();
    Type valueType = value.getType();
    if (isa<LWECiphertextType>(getElementTypeOrSelf(valueType))) {
      ImplicitLocOpBuilder b(validateOp.getLoc(), validateOp);
      SmallVector<NamedAttribute> attrs;
      // Transfer metadata from validateOp to CallOp
      attrs.push_back(b.getNamedAttr("debug.name", validateOp.getNameAttr()));
      if (validateOp.getMetadata()) {
        attrs.push_back(
            b.getNamedAttr("debug.metadata", validateOp.getMetadataAttr()));
      }
      attrs.push_back(b.getNamedAttr(
          "message.size", b.getStringAttr(std::to_string(messageSize))));

      auto debugFunc = getOrCreateExternalDebugFunc(module, lwePrivateKeyType,
                                                    valueType, typePairToInt);
      auto callOp =
          b.create<func::CallOp>(debugFunc, ArrayRef<Value>{privateKey, value});
      callOp->setDialectAttrs(attrs);

      validateOp.erase();
    }
    return WalkResult::advance();
  });

  return walkResult.wasInterrupted() ? failure() : success();
}

bool hasValidationOps(func::FuncOp op) {
  bool has = false;
  op.walk([&](debug::ValidateOp) {
    has = true;
    return WalkResult::interrupt();
  });
  return has;
}

struct AddDebugPort : impl::AddDebugPortBase<AddDebugPort> {
  using AddDebugPortBase::AddDebugPortBase;

  llvm::DenseMap<std::pair<Type, Type>, int> typePairToInt;

  void runOnOperation() override {
    ModuleOp module = cast<ModuleOp>(getOperation());
    llvm::DenseMap<func::FuncOp, Type> funcToKeyType;
    SmallVector<func::FuncOp, 16> worklist;

    SymbolTable symbolTable(module);
    llvm::DenseMap<func::FuncOp, SmallVector<func::CallOp, 4>> calleeToCalls;
    llvm::DenseSet<func::FuncOp> modifiedFuncs;

    // Step 1: Identify initial targets
    func::FuncOp entryFunc;
    if (!entryFunction.empty()) {
      entryFunc = symbolTable.lookup<func::FuncOp>(entryFunction);
    }

    if (entryFunc) {
      auto type = getPrivateKeyType(entryFunc);
      if (succeeded(type)) {
        funcToKeyType[entryFunc] = *type;
        worklist.push_back(entryFunc);
      } else if (hasValidationOps(entryFunc)) {
        entryFunc.emitError(
            "Cannot infer LWE private key type for entry function");
        signalPassFailure();
        return;
      }
    } else {
      for (auto funcOp : module.getOps<func::FuncOp>()) {
        if (funcOp.isExternal()) continue;

        bool shouldProcess = hasValidationOps(funcOp);
        if (!shouldProcess && insertDebugAfterEveryOp) {
          shouldProcess = succeeded(getPrivateKeyType(funcOp));
        }

        if (shouldProcess) {
          auto type = getPrivateKeyType(funcOp);
          if (succeeded(type)) {
            funcToKeyType[funcOp] = *type;
            worklist.push_back(funcOp);
          } else if (hasValidationOps(funcOp)) {
            funcOp.emitError(
                "Cannot infer LWE private key type for function with "
                "validation ops");
            signalPassFailure();
            return;
          }
        }
      }
    }

    // Step 2: Propagate
    while (!worklist.empty()) {
      func::FuncOp currentFunc = worklist.back();
      worklist.pop_back();
      Type keyType = funcToKeyType[currentFunc];

      auto symbolUses = SymbolTable::getSymbolUses(currentFunc, module);
      if (symbolUses) {
        for (auto use : *symbolUses) {
          Operation* user = use.getUser();
          auto callOp = dyn_cast<func::CallOp>(user);
          if (!callOp) continue;

          calleeToCalls[currentFunc].push_back(callOp);

          func::FuncOp caller = callOp->getParentOfType<func::FuncOp>();
          if (!caller) continue;

          if (funcToKeyType.find(caller) == funcToKeyType.end()) {
            funcToKeyType[caller] = keyType;
            worklist.push_back(caller);
          } else if (funcToKeyType[caller] != keyType) {
            caller.emitError("Conflicting LWE private key types required");
            signalPassFailure();
            return;
          }
        }
      }
    }

    // Step 3: Insert validation ops if requested
    if (insertDebugAfterEveryOp) {
      for (auto& pair : funcToKeyType) {
        insertValidationOps(pair.first);
      }
    }

    // Step 4: Add arguments
    for (auto& pair : funcToKeyType) {
      func::FuncOp funcOp = pair.first;
      Type keyType = pair.second;

      bool hasKey = false;
      for (auto arg : funcOp.getArguments()) {
        if (arg.getType() == keyType) {
          hasKey = true;
          break;
        }
      }

      if (!hasKey) {
        if (failed(
                funcOp.insertArgument(0, keyType, nullptr, funcOp.getLoc()))) {
          funcOp.emitError("failed to insert private key argument");
          signalPassFailure();
          return;
        }
        modifiedFuncs.insert(funcOp);
      }
    }

    // Step 5: Update call sites
    for (auto funcOp : modifiedFuncs) {
      for (auto callOp : calleeToCalls[funcOp]) {
        auto callerFunc = callOp->getParentOfType<func::FuncOp>();
        Type keyType = funcToKeyType[funcOp];

        Value keyToPass;
        for (auto arg : callerFunc.getArguments()) {
          if (arg.getType() == keyType) {
            keyToPass = arg;
            break;
          }
        }

        if (!keyToPass) {
          callOp.emitError(
              "Caller does not have the required LWE private key argument");
          signalPassFailure();
          return;
        }

        SmallVector<Value, 4> operands;
        operands.push_back(keyToPass);
        operands.append(callOp.getOperands().begin(),
                        callOp.getOperands().end());

        OpBuilder b(callOp);
        auto newCall =
            b.create<func::CallOp>(callOp.getLoc(), funcOp, operands);
        newCall->setDialectAttrs(callOp->getDialectAttrs());
        for (unsigned i = 0; i < callOp.getNumResults(); ++i) {
          callOp.getResult(i).replaceAllUsesWith(newCall.getResult(i));
        }
        callOp.erase();
      }
    }

    // Step 6: Lower validation ops
    for (auto& pair : funcToKeyType) {
      func::FuncOp funcOp = pair.first;
      Type keyType = pair.second;
      Value privateKey;
      for (auto arg : funcOp.getArguments()) {
        if (arg.getType() == keyType) {
          privateKey = arg;
          break;
        }
      }

      if (privateKey) {
        if (failed(lowerValidationOps(funcOp, privateKey, messageSize,
                                      typePairToInt))) {
          funcOp.emitError("failed to lower validation ops");
          signalPassFailure();
          return;
        }
      }
    }
  }
};
}  // namespace lwe
}  // namespace heir
}  // namespace mlir
