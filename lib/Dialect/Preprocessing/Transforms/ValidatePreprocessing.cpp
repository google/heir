#include "lib/Dialect/Preprocessing/Transforms/ValidatePreprocessing.h"

#include <cstddef>
#include <cstdint>

#include "lib/Dialect/Preprocessing/IR/PreprocessingOps.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingTypes.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"     // from @llvm-project
#include "llvm/include/llvm/ADT/DenseMap.h"     // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"      // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace preprocessing {

#define GEN_PASS_DEF_VALIDATEPREPROCESSING
#include "lib/Dialect/Preprocessing/Transforms/Passes.h.inc"

struct ValidatePreprocessing
    : impl::ValidatePreprocessingBase<ValidatePreprocessing> {
  using ValidatePreprocessingBase::ValidatePreprocessingBase;

  void runOnOperation() override {
    SmallVector<EmptyOp, 2> emptyOps;
    DenseMap<uint32_t, SmallVector<StoreOp, 2>> storesBySite;
    DenseMap<uint32_t, SmallVector<LoadOp, 2>> loadsBySite;

    getOperation()->walk([&](Operation* op) {
      if (auto emptyOp = dyn_cast<EmptyOp>(op)) {
        emptyOps.push_back(emptyOp);
      } else if (auto storeOp = dyn_cast<StoreOp>(op)) {
        storesBySite[storeOp.getSiteId()].push_back(storeOp);
      } else if (auto loadOp = dyn_cast<LoadOp>(op)) {
        loadsBySite[loadOp.getSiteId()].push_back(loadOp);
      }
    });

    if (emptyOps.size() > 1) {
      for (size_t i = 1; i < emptyOps.size(); ++i) {
        auto diag = emptyOps[i]->emitOpError()
                    << "more than one preprocessing.empty allocation in module";
        diag.attachNote(emptyOps[0]->getLoc()) << "previous allocation here";
      }
      signalPassFailure();
      // Do NOT return, continue to site pairing checks
    }

    SmallVector<uint32_t, 4> siteIds;
    for (const auto& [siteId, stores] : storesBySite) {
      siteIds.push_back(siteId);
    }
    for (const auto& [siteId, loads] : loadsBySite) {
      if (!storesBySite.count(siteId)) {
        siteIds.push_back(siteId);
      }
    }
    llvm::sort(siteIds);

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
          dyn_cast<PreprocessingStorageType>(store.getStorage().getType());
      auto loadStorageTy =
          dyn_cast<PreprocessingStorageType>(load.getStorage().getType());

      if (store.getElementType() != load.getElementType()) {
        auto diag = store->emitOpError()
                    << "store element type " << store.getElementType()
                    << " does not match load element type "
                    << load.getElementType();
        diag.attachNote(load->getLoc()) << "corresponding load here";
        signalPassFailure();
        continue;
      }

      if (storeStorageTy && loadStorageTy && storeStorageTy != loadStorageTy) {
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
