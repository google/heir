#include "lib/Transforms/ExternalizeConstants/ExternalizeConstants.h"

#include <string>
#include <system_error>
#include <vector>

#include "lib/Dialect/Preprocessing/IR/PreprocessingOps.h"
#include "llvm/include/llvm/Support/FileSystem.h"      // from @llvm-project
#include "llvm/include/llvm/Support/MD5.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Path.h"            // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"         // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"            // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"      // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_EXTERNALIZECONSTANTS
#include "lib/Transforms/ExternalizeConstants/ExternalizeConstants.h.inc"

struct ExternalizeConstants
    : impl::ExternalizeConstantsBase<ExternalizeConstants> {
  using ExternalizeConstantsBase::ExternalizeConstantsBase;

  void runOnOperation() override {
    Operation* op = getOperation();
    IRRewriter rewriter(&getContext());

    op->walk([&](arith::ConstantOp constantOp) -> WalkResult {
      auto tensorType = dyn_cast<RankedTensorType>(constantOp.getType());
      if (!tensorType) return WalkResult::advance();

      auto valueAttr = constantOp.getValue();
      ArrayRef<char> rawData;
      std::vector<char> unpackedI1Data;
      int64_t numElements = 0;

      if (auto denseAttr = dyn_cast<DenseElementsAttr>(valueAttr)) {
        numElements = denseAttr.getNumElements();
        if (numElements < thresholdElements) return WalkResult::advance();

        // A splat is O(1) in the IR, and the emitters already expand it in place
        // (a fill constructor in C++, slices.Repeat in Go), so there is nothing
        // to gain by writing numElements copies to disk.
        if (denseAttr.isSplat()) return WalkResult::advance();

        if (tensorType.getElementType().isInteger(1)) {
          unpackedI1Data.reserve(numElements);
          for (bool val : denseAttr.getValues<bool>()) {
            unpackedI1Data.push_back(val ? 1 : 0);
          }
          rawData =
              ArrayRef<char>(unpackedI1Data.data(), unpackedI1Data.size());
        } else {
          rawData = denseAttr.getRawData();
        }
      } else if (auto denseResourceAttr =
                     dyn_cast<DenseResourceElementsAttr>(valueAttr)) {
        numElements = denseResourceAttr.getNumElements();
        if (numElements < thresholdElements) return WalkResult::advance();

        if (tensorType.getElementType().isInteger(1)) {
          if (auto boolResourceAttr =
                  dyn_cast<DenseBoolResourceElementsAttr>(denseResourceAttr)) {
            if (auto maybeArray = boolResourceAttr.tryGetAsArrayRef()) {
              unpackedI1Data.reserve(numElements);
              for (bool val : *maybeArray) {
                unpackedI1Data.push_back(val ? 1 : 0);
              }
              rawData =
                  ArrayRef<char>(unpackedI1Data.data(), unpackedI1Data.size());
            } else {
              return constantOp->emitError("Failed to get bool resource data"),
                     WalkResult::interrupt();
            }
          } else {
            return constantOp->emitError("Expected bool resource attr"),
                   WalkResult::interrupt();
          }
        } else {
          rawData = denseResourceAttr.getData();
          if (rawData.empty()) {
            return constantOp->emitError(
                       "Dense resource has no data (possibly elided)"),
                   WalkResult::interrupt();
          }
        }
      } else {
        return WalkResult::advance();
      }

      // Compute MD5
      llvm::MD5 hash;
      hash.update(llvm::StringRef(rawData.data(), rawData.size()));
      llvm::MD5::MD5Result result;
      hash.final(result);
      llvm::SmallString<32> hexHash;
      llvm::MD5::stringifyResult(result, hexHash);

      std::string fileName =
          (llvm::Twine("constant_") + hexHash + ".bin").str();

      // Write to output-dir
      llvm::SmallString<128> outputPath(outputDir);
      llvm::sys::path::append(outputPath, fileName);

      std::error_code ec;
      llvm::raw_fd_ostream os(outputPath.str(), ec, llvm::sys::fs::OF_None);
      if (ec) {
        constantOp->emitError()
            << "Failed to open file for writing: " << outputPath.str()
            << " error: " << ec.message();
        signalPassFailure();
        return WalkResult::interrupt();
      }
      os.write(rawData.data(), rawData.size());
      os.close();

      // Replace with preprocessing.load_resource
      llvm::SmallString<128> runtimePath(runtimeLoadDir);
      llvm::sys::path::append(runtimePath, fileName);

      rewriter.setInsertionPoint(constantOp);
      auto loadOp = preprocessing::LoadResourceOp::create(
          rewriter, constantOp.getLoc(), tensorType,
          rewriter.getStringAttr(runtimePath.str()));

      rewriter.replaceOp(constantOp, loadOp.getResult());
      return WalkResult::advance();
    });
  }
};

}  // namespace heir
}  // namespace mlir
