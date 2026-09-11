#include "lib/Transforms/GatherZeroEncryptions/GatherZeroEncryptions.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <string>

#include "lib/Dialect/ModuleAttributes.h"
#include "llvm/include/llvm/ADT/DenseMap.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_GATHERZEROENCRYPTIONS
#include "lib/Transforms/GatherZeroEncryptions/GatherZeroEncryptions.h.inc"

namespace {

struct ZeroArgInfo {
  unsigned oldArgIdx;
  int64_t index;
  BlockArgument arg;
  Type type;
};

struct GatheredFuncInfo {
  SmallVector<unsigned> oldZeroArgIndices;
  unsigned firstZeroArgIdx;
  Value newMemrefArg;
};

}  // namespace

struct GatherZeroEncryptions
    : impl::GatherZeroEncryptionsBase<GatherZeroEncryptions> {
  using GatherZeroEncryptionsBase::GatherZeroEncryptionsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext* context = &getContext();

    // Step 1: For each non-client helper function, collect all arguments with
    // client.enc_zero_arg, sorted by index. Replace them with a single
    // memref<N x !ct> argument.
    DenseMap<Operation*, GatheredFuncInfo> gatheredFuncs;

    for (auto func : module.getOps<func::FuncOp>()) {
      if (func.isDeclaration() || func->hasAttr(kClientEncZeroFuncAttrName)) {
        continue;
      }

      SmallVector<ZeroArgInfo> zeroArgs;
      for (BlockArgument arg : func.getArguments()) {
        unsigned argNum = arg.getArgNumber();
        if (auto dictAttr = func.getArgAttrOfType<DictionaryAttr>(
                argNum, kClientEncZeroArgAttrName)) {
          int64_t index = 0;
          if (auto idxAttr = dictAttr.getAs<IntegerAttr>(kClientHelperIndex)) {
            index = idxAttr.getInt();
          }
          zeroArgs.push_back({argNum, index, arg, arg.getType()});
        } else if (func.getArgAttr(argNum, kClientEncZeroArgAttrName)) {
          zeroArgs.push_back({argNum, 0, arg, arg.getType()});
        }
      }

      if (zeroArgs.empty()) {
        continue;
      }

      llvm::sort(zeroArgs, [](const ZeroArgInfo& a, const ZeroArgInfo& b) {
        return a.index < b.index;
      });

      int64_t n = zeroArgs.size();
      Type ctTy = zeroArgs[0].type;
      MemRefType memrefType = MemRefType::get({n}, ctTy);

      SmallVector<unsigned> oldZeroArgIndices;
      unsigned firstZeroArgIdx = func.getNumArguments();
      for (const auto& info : zeroArgs) {
        firstZeroArgIdx = std::min(firstZeroArgIdx, info.oldArgIdx);
        oldZeroArgIndices.push_back(info.oldArgIdx);
      }

      // Build new argument types and attribute dictionaries
      SmallVector<Type> newArgTypes;
      SmallVector<DictionaryAttr> newArgAttrs;
      unsigned numArgs = func.getNumArguments();
      for (unsigned i = 0; i < numArgs; ++i) {
        bool isZeroArg = llvm::is_contained(oldZeroArgIndices, i);
        if (!isZeroArg) {
          newArgTypes.push_back(func.getArgument(i).getType());
          DictionaryAttr oldDict = func.getArgAttrDict(i);
          newArgAttrs.push_back(oldDict ? oldDict
                                        : DictionaryAttr::get(context, {}));
        } else if (i == firstZeroArgIdx) {
          newArgTypes.push_back(memrefType);
          NamedAttribute unitAttr(
              StringAttr::get(context, kClientEncZeroArgAttrName),
              UnitAttr::get(context));
          newArgAttrs.push_back(DictionaryAttr::get(context, unitAttr));
        }
      }

      func.setType(
          FunctionType::get(context, newArgTypes, func.getResultTypes()));
      func.setAllArgAttrs(newArgAttrs);

      Block* entryBlock = &func.getBody().front();
      BlockArgument newMemrefArg = entryBlock->insertArgument(
          firstZeroArgIdx, memrefType, func.getLoc());

      OpBuilder builder(entryBlock, entryBlock->begin());
      for (int64_t k = 0; k < n; ++k) {
        Value idxVal = builder.create<arith::ConstantIndexOp>(func.getLoc(), k);
        Value loaded = builder.create<memref::LoadOp>(
            func.getLoc(), newMemrefArg, ValueRange{idxVal});
        zeroArgs[k].arg.replaceAllUsesWith(loaded);
      }

      SmallVector<unsigned> currentIndicesToErase;
      for (unsigned oldIdx : oldZeroArgIndices) {
        if (oldIdx >= firstZeroArgIdx) {
          currentIndicesToErase.push_back(oldIdx + 1);
        } else {
          currentIndicesToErase.push_back(oldIdx);
        }
      }
      llvm::sort(currentIndicesToErase, std::greater<unsigned>());
      for (unsigned idx : currentIndicesToErase) {
        entryBlock->eraseArgument(idx);
      }

      gatheredFuncs[func.getOperation()] =
          GatheredFuncInfo{oldZeroArgIndices, firstZeroArgIdx, newMemrefArg};
    }

    // Step 2: Update internal calls (e.g. if @main calls @main__preprocessed)
    SmallVector<func::CallOp> callsToUpdate;
    module.walk([&](func::CallOp callOp) {
      auto callee = module.lookupSymbol<func::FuncOp>(callOp.getCallee());
      if (callee && gatheredFuncs.count(callee.getOperation())) {
        callsToUpdate.push_back(callOp);
      }
    });

    for (func::CallOp callOp : callsToUpdate) {
      auto callee = module.lookupSymbol<func::FuncOp>(callOp.getCallee());
      const GatheredFuncInfo& calleeInfo = gatheredFuncs[callee.getOperation()];
      auto caller = callOp->getParentOfType<func::FuncOp>();

      Value memrefToPass;
      auto callerIt = gatheredFuncs.find(caller.getOperation());
      if (callerIt != gatheredFuncs.end()) {
        memrefToPass = callerIt->second.newMemrefArg;
      } else {
        for (auto arg : caller.getArguments()) {
          if (caller.getArgAttr(arg.getArgNumber(),
                                kClientEncZeroArgAttrName)) {
            memrefToPass = arg;
            break;
          }
        }
      }
      assert(memrefToPass &&
             "Caller must have a gathered memref argument to forward");

      OpBuilder builder(callOp);
      SmallVector<Value> newOperands;
      for (unsigned i = 0; i < callOp.getNumOperands(); ++i) {
        bool isZeroArg = llvm::is_contained(calleeInfo.oldZeroArgIndices, i);
        if (!isZeroArg) {
          newOperands.push_back(callOp.getOperand(i));
        } else if (i == calleeInfo.firstZeroArgIdx) {
          newOperands.push_back(memrefToPass);
        }
      }

      auto newCallOp =
          builder.create<func::CallOp>(callOp.getLoc(), callOp.getCallee(),
                                       callOp.getResultTypes(), newOperands);
      callOp.replaceAllUsesWith(newCallOp);
      callOp.erase();
    }

    // Step 3: Client helpers: group by target function, create a single
    // combined helper @<func>__encrypt__zeros(...) -> memref<N x !ct>.
    DenseMap<StringRef, SmallVector<func::FuncOp>> helpersByTarget;
    module.walk([&](func::FuncOp func) {
      if (func->hasAttr(kClientEncZeroFuncAttrName)) {
        StringRef targetFunc;
        if (auto dict = func->getAttrOfType<DictionaryAttr>(
                kClientEncZeroFuncAttrName)) {
          if (auto funcNameAttr =
                  dict.getAs<StringAttr>(kClientHelperFuncName)) {
            targetFunc = funcNameAttr.getValue();
          }
        }
        if (targetFunc.empty()) {
          StringRef symName = func.getSymName();
          auto pos = symName.find("__encrypt__zero__");
          if (pos != StringRef::npos) {
            targetFunc = symName.substr(0, pos);
          }
        }
        if (!targetFunc.empty()) {
          helpersByTarget[targetFunc].push_back(func);
        }
      }
    });

    for (auto& [targetFunc, helpers] : helpersByTarget) {
      if (helpers.empty()) continue;

      auto getHelperIndex = [](func::FuncOp func) -> int64_t {
        if (auto dict = func->getAttrOfType<DictionaryAttr>(
                kClientEncZeroFuncAttrName)) {
          if (auto idxAttr = dict.getAs<IntegerAttr>(kClientHelperIndex)) {
            return idxAttr.getInt();
          }
        }
        StringRef symName = func.getSymName();
        auto pos = symName.rfind("__encrypt__zero__");
        if (pos != StringRef::npos) {
          int64_t idx = 0;
          if (!symName.substr(pos + 17).getAsInteger(10, idx)) {
            return idx;
          }
        }
        return 0;
      };

      llvm::sort(helpers, [&](func::FuncOp a, func::FuncOp b) {
        return getHelperIndex(a) < getHelperIndex(b);
      });

      int64_t n = helpers.size();
      Type ctTy = helpers[0].getResultTypes()[0];
      MemRefType memrefType = MemRefType::get({n}, ctTy);

      std::string combinedName = (targetFunc + "__encrypt__zeros").str();
      FunctionType funcType = FunctionType::get(
          context, helpers[0].getArgumentTypes(), {memrefType});

      OpBuilder builder(module.getContext());
      builder.setInsertionPointAfter(helpers.back());
      auto combinedFunc = builder.create<func::FuncOp>(helpers[0].getLoc(),
                                                       combinedName, funcType);
      combinedFunc.setVisibility(helpers[0].getVisibility());

      combinedFunc->setAttr(
          kClientEncZeroFuncAttrName,
          builder.getDictionaryAttr({
              builder.getNamedAttr(kClientHelperFuncName,
                                   builder.getStringAttr(targetFunc)),
          }));

      for (unsigned i = 0; i < helpers[0].getNumArguments(); ++i) {
        if (auto attrs = helpers[0].getArgAttrDict(i)) {
          combinedFunc.setArgAttrs(i, attrs);
        }
      }

      Block* entryBlock = combinedFunc.addEntryBlock();
      builder.setInsertionPointToEnd(entryBlock);

      Location loc = combinedFunc.getLoc();
      Value allocated = builder.create<memref::AllocOp>(loc, memrefType);

      for (auto [k, helper] : llvm::enumerate(helpers)) {
        IRMapping map;
        for (unsigned argIdx = 0; argIdx < helper.getNumArguments(); ++argIdx) {
          map.map(helper.getArgument(argIdx), combinedFunc.getArgument(argIdx));
        }

        for (auto& op : helper.getBody().front()) {
          if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
            Value retVal = map.lookup(returnOp.getOperand(0));
            Value idxVal = builder.create<arith::ConstantIndexOp>(loc, k);
            builder.create<memref::StoreOp>(loc, retVal, allocated,
                                            ValueRange{idxVal});
          } else {
            builder.clone(op, map);
          }
        }
      }

      builder.create<func::ReturnOp>(loc, ValueRange{allocated});

      for (auto helper : helpers) {
        helper.erase();
      }
    }
  }
};

}  // namespace heir
}  // namespace mlir
