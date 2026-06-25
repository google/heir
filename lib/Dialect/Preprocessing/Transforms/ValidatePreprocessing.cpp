#include "lib/Dialect/Preprocessing/Transforms/ValidatePreprocessing.h"

#include <cstddef>
#include <cstdint>
#include <set>

#include "lib/Dialect/Preprocessing/IR/PreprocessingOps.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingTypes.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/DenseMap.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"          // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

namespace mlir {
namespace heir {
namespace preprocessing {

#define GEN_PASS_DEF_VALIDATEPREPROCESSING
#include "lib/Dialect/Preprocessing/Transforms/Passes.h.inc"

struct ValidatePreprocessing
    : impl::ValidatePreprocessingBase<ValidatePreprocessing> {
  using ValidatePreprocessingBase::ValidatePreprocessingBase;

  void runOnOperation() override {
    Operation* module = getOperation();

    bool hasMultipleEmpties = false;
    module->walk([&](func::FuncOp funcOp) {
      SmallVector<EmptyOp> emptyOps(funcOp.getOps<EmptyOp>());
      if (emptyOps.size() > 1) {
        for (size_t i = 1; i < emptyOps.size(); ++i) {
          auto diag =
              emptyOps[i]->emitOpError()
              << "more than one preprocessing.empty allocation in function";
          diag.attachNote(emptyOps[0]->getLoc()) << "previous allocation here";
        }
        hasMultipleEmpties = true;
      }
    });
    if (hasMultipleEmpties) {
      signalPassFailure();
    }

    DenseMap<uint32_t, SmallVector<StoreOp, 2>> storesBySite;
    DenseMap<uint32_t, SmallVector<LoadOp, 2>> loadsBySite;

    module->walk([&](Operation* op) {
      if (auto storeOp = dyn_cast<StoreOp>(op)) {
        storesBySite[storeOp.getSiteId()].push_back(storeOp);
      } else if (auto loadOp = dyn_cast<LoadOp>(op)) {
        loadsBySite[loadOp.getSiteId()].push_back(loadOp);
      }
    });

    std::set<uint32_t> siteIds;
    for (const auto& [siteId, stores] : storesBySite) {
      siteIds.insert(siteId);
    }
    for (const auto& [siteId, loads] : loadsBySite) {
      siteIds.insert(siteId);
    }

    for (uint32_t siteId : siteIds) {
      ArrayRef<StoreOp> stores;
      auto storesIt = storesBySite.find(siteId);
      if (storesIt != storesBySite.end()) {
        stores = storesIt->second;
      }

      ArrayRef<LoadOp> loads;
      auto loadsIt = loadsBySite.find(siteId);
      if (loadsIt != loadsBySite.end()) {
        loads = loadsIt->second;
      }

      if (stores.size() != 1 || loads.size() != 1) {
        if (stores.empty() && !loads.empty()) {
          for (auto load : loads) {
            load->emitOpError()
                << "lacks a corresponding store for site_id: " << siteId;
          }
        } else if (loads.empty() && !stores.empty()) {
          for (auto store : stores) {
            store->emitOpError()
                << "lacks a corresponding load for site_id: " << siteId;
          }
        } else {
          if (stores.size() > 1) {
            for (size_t i = 1; i < stores.size(); ++i) {
              auto diag = stores[i]->emitOpError()
                          << "duplicate store for site_id: " << siteId;
              diag.attachNote(stores[0]->getLoc()) << "previous store here";
            }
          }
          if (loads.size() > 1) {
            for (size_t i = 1; i < loads.size(); ++i) {
              auto diag = loads[i]->emitOpError()
                          << "duplicate load for site_id: " << siteId;
              diag.attachNote(loads[0]->getLoc()) << "previous load here";
            }
          }
        }
        signalPassFailure();
        continue;
      }

      StoreOp store = stores[0];
      LoadOp load = loads[0];

      if (store.getIndices().size() != load.getIndices().size()) {
        auto diag = store->emitOpError()
                    << "store and load index arity mismatch for site_id: "
                    << siteId;
        diag.attachNote(load->getLoc()) << "corresponding load here";
        signalPassFailure();
        continue;
      }

      auto storeStorageTy =
          cast<PreprocessingStorageType>(store.getStorage().getType());
      auto loadStorageTy =
          cast<PreprocessingStorageType>(load.getStorage().getType());

      if (store.getElementType() != load.getElementType()) {
        auto diag = store->emitOpError()
                    << "store element type " << store.getElementType()
                    << " does not match load element type "
                    << load.getElementType();
        diag.attachNote(load->getLoc()) << "corresponding load here";
        signalPassFailure();
        continue;
      }

      if (storeStorageTy != loadStorageTy) {
        auto diag = store->emitOpError()
                    << "store storage type " << storeStorageTy
                    << " does not match load storage type " << loadStorageTy;
        diag.attachNote(load->getLoc()) << "corresponding load here";
        signalPassFailure();
        continue;
      }
    }
  }
};

}  // namespace preprocessing
}  // namespace heir
}  // namespace mlir
