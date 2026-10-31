#include "lib/Dialect/CheddarGPU/Conversions/CheddarGPUToGPU/ModDownEpilogueKernels.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

#include "lib/Dialect/CheddarGPU/Conversions/CheddarGPUToGPU/KernelBuilders.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUAttributes.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUOps.h"
#include "llvm/include/llvm/Support/FormatVariadic.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"         // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"    // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                   // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"      // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar_gpu {

namespace {

constexpr int64_t kBlockDim = 256;

struct ConvertModDownEpilogueOp : public OpRewritePattern<ModDownEpilogueOp> {
  using OpRewritePattern<ModDownEpilogueOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ModDownEpilogueOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams) {
      return op.emitOpError("missing cheddar_gpu.params on module");
    }

    uint32_t word = modParams->word;
    int64_t degree = modParams->degree;

    auto outType = cast<MemRefType>(op.getOut().getType());
    int64_t numLimbs = outType.getDimSize(0);

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto module = op->getParentOfType<ModuleOp>();
    gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);

    Type intType = IntegerType::get(b.getContext(), word);
    Type indexType = b.getIndexType();
    auto dynMemref2DType =
        MemRefType::get({ShapedType::kDynamic, degree}, intType);
    auto dynMemref1DType = MemRefType::get({ShapedType::kDynamic}, intType);

    bool hasSrc2 = (op.getSrc2() != nullptr);
    bool hasPadding = (op.getSrc2Padding() != nullptr);

    int64_t blockSize = std::min<int64_t>(kBlockDim, degree);
    if (blockSize < 1) blockSize = 1;
    int64_t gridDimX = (degree + blockSize - 1) / blockSize;
    int64_t gridDimY = numLimbs;

    Value c1 = b.create<arith::ConstantIndexOp>(1);
    Value gridX = b.create<arith::ConstantIndexOp>(gridDimX);
    Value gridY = b.create<arith::ConstantIndexOp>(gridDimY);
    Value blockX = b.create<arith::ConstantIndexOp>(blockSize);

    gpu::KernelDim3 gridSize{gridX, gridY, c1};
    gpu::KernelDim3 blockDim{blockX, c1, c1};

    if (!hasSrc2) {
      // Variant 1: No src2
      std::string kernelName = llvm::formatv(
          "cheddar_gpu_mod_down_epilogue_nosrc2_w{0}_deg{1}", word, degree);
      auto kernelType = FunctionType::get(
          b.getContext(), {dynMemref2DType, dynMemref2DType, dynMemref1DType},
          {});

      gpu::GPUFuncOp kernelFunc = getOrCreateKernel(
          gpuMod, kernelName, kernelType,
          [&](OpBuilder& kb, Location loc, Block* entryBlock) {
            ImplicitLocOpBuilder ib(loc, kb);
            Value outArg = entryBlock->getArgument(0);
            Value nttArg = entryBlock->getArgument(1);
            Value invPProdArg = entryBlock->getArgument(2);

            Value bx = ib.create<gpu::BlockIdOp>(gpu::Dimension::x);
            Value by = ib.create<gpu::BlockIdOp>(gpu::Dimension::y);
            Value tx = ib.create<gpu::ThreadIdOp>(gpu::Dimension::x);
            Value bdim = ib.create<gpu::BlockDimOp>(gpu::Dimension::x);
            Value bxDim = ib.create<arith::MulIOp>(bx, bdim);
            Value coeffIdx = ib.create<arith::AddIOp>(bxDim, tx);

            if (degree % blockSize != 0) {
              Value cDegree = ib.create<arith::ConstantIndexOp>(degree);
              Value inBounds = ib.create<arith::CmpIOp>(
                  arith::CmpIPredicate::ult, coeffIdx, cDegree);
              auto ifOp =
                  ib.create<scf::IfOp>(inBounds, /*withElseRegion=*/false);
              ib.setInsertionPointToStart(&ifOp.getThenRegion().front());
            }

            auto [primes, qInv] = getDevicePrimeGlobals(ib, word);
            Value qVal = ib.create<memref::LoadOp>(primes, ValueRange{by});
            Value qInvVal = ib.create<memref::LoadOp>(qInv, ValueRange{by});
            Value nttVal =
                ib.create<memref::LoadOp>(nttArg, ValueRange{by, coeffIdx});
            Value invPProdVal =
                ib.create<memref::LoadOp>(invPProdArg, ValueRange{by});

            Value zero = ib.create<arith::ConstantIntOp>(intType, 0);
            Value diff = buildModSubRuntime(ib, zero, nttVal, qVal);
            Value res =
                buildMontMulRuntime(ib, diff, invPProdVal, qVal, qInvVal);

            ib.create<memref::StoreOp>(res, outArg, ValueRange{by, coeffIdx});
            ib.create<gpu::ReturnOp>();
          });

      Value outDyn = castToDynamicFirstDim(b, op.getLoc(), op.getOut());
      Value nttDyn = castToDynamicFirstDim(b, op.getLoc(), op.getNttBconv());
      Value invPProdDyn = castToDynamic1D(b, op.getLoc(), op.getInvPrimeProd());

      emitKernelLaunch(b, op.getLoc(), kernelFunc, gridSize, blockDim,
                       ValueRange{outDyn, nttDyn, invPProdDyn});
    } else if (!hasPadding) {
      // Variant 2: Has src2, no padding
      Value src2Dyn = castToDynamicFirstDim(b, op.getLoc(), op.getSrc2());
      Type src2Type = src2Dyn.getType();
      std::string kernelName = llvm::formatv(
          "cheddar_gpu_mod_down_epilogue_src2_w{0}_deg{1}", word, degree);
      if (auto memref = dyn_cast<MemRefType>(src2Type)) {
        if (auto layout = dyn_cast<StridedLayoutAttr>(memref.getLayout())) {
          if (layout.getOffset() != 0) {
            kernelName += llvm::formatv("_off{0}", layout.getOffset());
          }
        }
      }
      auto kernelType =
          FunctionType::get(b.getContext(),
                            {dynMemref2DType, dynMemref2DType, dynMemref1DType,
                             src2Type, indexType, indexType},
                            {});

      gpu::GPUFuncOp kernelFunc = getOrCreateKernel(
          gpuMod, kernelName, kernelType,
          [&](OpBuilder& kb, Location loc, Block* entryBlock) {
            ImplicitLocOpBuilder ib(loc, kb);
            Value outArg = entryBlock->getArgument(0);
            Value nttArg = entryBlock->getArgument(1);
            Value invPProdArg = entryBlock->getArgument(2);
            Value src2Arg = entryBlock->getArgument(3);
            Value padStartArg = entryBlock->getArgument(4);
            Value padEndArg = entryBlock->getArgument(5);

            Value bx = ib.create<gpu::BlockIdOp>(gpu::Dimension::x);
            Value by = ib.create<gpu::BlockIdOp>(gpu::Dimension::y);
            Value tx = ib.create<gpu::ThreadIdOp>(gpu::Dimension::x);
            Value bdim = ib.create<gpu::BlockDimOp>(gpu::Dimension::x);
            Value bxDim = ib.create<arith::MulIOp>(bx, bdim);
            Value coeffIdx = ib.create<arith::AddIOp>(bxDim, tx);

            if (degree % blockSize != 0) {
              Value cDegree = ib.create<arith::ConstantIndexOp>(degree);
              Value inBounds = ib.create<arith::CmpIOp>(
                  arith::CmpIPredicate::ult, coeffIdx, cDegree);
              auto ifOp =
                  ib.create<scf::IfOp>(inBounds, /*withElseRegion=*/false);
              ib.setInsertionPointToStart(&ifOp.getThenRegion().front());
            }

            auto [primes, qInv] = getDevicePrimeGlobals(ib, word);
            Value qVal = ib.create<memref::LoadOp>(primes, ValueRange{by});
            Value qInvVal = ib.create<memref::LoadOp>(qInv, ValueRange{by});
            Value nttVal =
                ib.create<memref::LoadOp>(nttArg, ValueRange{by, coeffIdx});
            Value invPProdVal =
                ib.create<memref::LoadOp>(invPProdArg, ValueRange{by});

            Value isGe = ib.create<arith::CmpIOp>(arith::CmpIPredicate::uge, by,
                                                  padStartArg);
            Value isLt = ib.create<arith::CmpIOp>(arith::CmpIPredicate::ult, by,
                                                  padEndArg);
            Value inPad = ib.create<arith::AndIOp>(isGe, isLt);

            auto ifPad = ib.create<scf::IfOp>(TypeRange{intType}, inPad,
                                              /*withElseRegion=*/true);
            ib.setInsertionPointToStart(&ifPad.getThenRegion().front());
            Value s2Limb = ib.create<arith::SubIOp>(by, padStartArg);
            Value s2Val = ib.create<memref::LoadOp>(
                src2Arg, ValueRange{s2Limb, coeffIdx});
            ib.create<scf::YieldOp>(ValueRange{s2Val});

            ib.setInsertionPointToStart(&ifPad.getElseRegion().front());
            Value zero = ib.create<arith::ConstantIntOp>(intType, 0);
            ib.create<scf::YieldOp>(ValueRange{zero});

            ib.setInsertionPointAfter(ifPad);
            Value term1 = ifPad.getResult(0);
            Value diff = buildModSubRuntime(ib, term1, nttVal, qVal);
            Value res =
                buildMontMulRuntime(ib, diff, invPProdVal, qVal, qInvVal);

            ib.create<memref::StoreOp>(res, outArg, ValueRange{by, coeffIdx});
            ib.create<gpu::ReturnOp>();
          });

      Value outDyn = castToDynamicFirstDim(b, op.getLoc(), op.getOut());
      Value nttDyn = castToDynamicFirstDim(b, op.getLoc(), op.getNttBconv());
      Value invPProdDyn = castToDynamic1D(b, op.getLoc(), op.getInvPrimeProd());
      Value padStartVal = b.create<arith::ConstantIndexOp>(op.getPadStart());
      Value padEndVal = b.create<arith::ConstantIndexOp>(op.getPadEnd());

      emitKernelLaunch(b, op.getLoc(), kernelFunc, gridSize, blockDim,
                       ValueRange{outDyn, nttDyn, invPProdDyn, src2Dyn,
                                  padStartVal, padEndVal});
    } else {
      // Variant 3: Has src2 and src2_padding
      Value src2Dyn = castToDynamicFirstDim(b, op.getLoc(), op.getSrc2());
      Type src2Type = src2Dyn.getType();
      std::string kernelName = llvm::formatv(
          "cheddar_gpu_mod_down_epilogue_src2_pad_w{0}_deg{1}", word, degree);
      if (auto memref = dyn_cast<MemRefType>(src2Type)) {
        if (auto layout = dyn_cast<StridedLayoutAttr>(memref.getLayout())) {
          if (layout.getOffset() != 0) {
            kernelName += llvm::formatv("_off{0}", layout.getOffset());
          }
        }
      }
      auto kernelType =
          FunctionType::get(b.getContext(),
                            {dynMemref2DType, dynMemref2DType, dynMemref1DType,
                             src2Type, dynMemref1DType, indexType, indexType},
                            {});

      gpu::GPUFuncOp kernelFunc = getOrCreateKernel(
          gpuMod, kernelName, kernelType,
          [&](OpBuilder& kb, Location loc, Block* entryBlock) {
            ImplicitLocOpBuilder ib(loc, kb);
            Value outArg = entryBlock->getArgument(0);
            Value nttArg = entryBlock->getArgument(1);
            Value invPProdArg = entryBlock->getArgument(2);
            Value src2Arg = entryBlock->getArgument(3);
            Value padArg = entryBlock->getArgument(4);
            Value padStartArg = entryBlock->getArgument(5);
            Value padEndArg = entryBlock->getArgument(6);

            Value bx = ib.create<gpu::BlockIdOp>(gpu::Dimension::x);
            Value by = ib.create<gpu::BlockIdOp>(gpu::Dimension::y);
            Value tx = ib.create<gpu::ThreadIdOp>(gpu::Dimension::x);
            Value bdim = ib.create<gpu::BlockDimOp>(gpu::Dimension::x);
            Value bxDim = ib.create<arith::MulIOp>(bx, bdim);
            Value coeffIdx = ib.create<arith::AddIOp>(bxDim, tx);

            if (degree % blockSize != 0) {
              Value cDegree = ib.create<arith::ConstantIndexOp>(degree);
              Value inBounds = ib.create<arith::CmpIOp>(
                  arith::CmpIPredicate::ult, coeffIdx, cDegree);
              auto ifOp =
                  ib.create<scf::IfOp>(inBounds, /*withElseRegion=*/false);
              ib.setInsertionPointToStart(&ifOp.getThenRegion().front());
            }

            auto [primes, qInv] = getDevicePrimeGlobals(ib, word);
            Value qVal = ib.create<memref::LoadOp>(primes, ValueRange{by});
            Value qInvVal = ib.create<memref::LoadOp>(qInv, ValueRange{by});
            Value nttVal =
                ib.create<memref::LoadOp>(nttArg, ValueRange{by, coeffIdx});
            Value invPProdVal =
                ib.create<memref::LoadOp>(invPProdArg, ValueRange{by});

            Value isGe = ib.create<arith::CmpIOp>(arith::CmpIPredicate::uge, by,
                                                  padStartArg);
            Value isLt = ib.create<arith::CmpIOp>(arith::CmpIPredicate::ult, by,
                                                  padEndArg);
            Value inPad = ib.create<arith::AndIOp>(isGe, isLt);

            auto ifPad = ib.create<scf::IfOp>(TypeRange{intType}, inPad,
                                              /*withElseRegion=*/true);
            ib.setInsertionPointToStart(&ifPad.getThenRegion().front());
            Value s2Limb = ib.create<arith::SubIOp>(by, padStartArg);
            Value s2Val = ib.create<memref::LoadOp>(
                src2Arg, ValueRange{s2Limb, coeffIdx});
            Value padVal =
                ib.create<memref::LoadOp>(padArg, ValueRange{s2Limb});
            Value pMul = buildMontMulRuntime(ib, s2Val, padVal, qVal, qInvVal);
            ib.create<scf::YieldOp>(ValueRange{pMul});

            ib.setInsertionPointToStart(&ifPad.getElseRegion().front());
            Value zero = ib.create<arith::ConstantIntOp>(intType, 0);
            ib.create<scf::YieldOp>(ValueRange{zero});

            ib.setInsertionPointAfter(ifPad);
            Value term1 = ifPad.getResult(0);
            Value diff = buildModSubRuntime(ib, term1, nttVal, qVal);
            Value res =
                buildMontMulRuntime(ib, diff, invPProdVal, qVal, qInvVal);

            ib.create<memref::StoreOp>(res, outArg, ValueRange{by, coeffIdx});
            ib.create<gpu::ReturnOp>();
          });

      Value outDyn = castToDynamicFirstDim(b, op.getLoc(), op.getOut());
      Value nttDyn = castToDynamicFirstDim(b, op.getLoc(), op.getNttBconv());
      Value invPProdDyn = castToDynamic1D(b, op.getLoc(), op.getInvPrimeProd());
      Value src2PadDyn = castToDynamic1D(b, op.getLoc(), op.getSrc2Padding());
      Value padStartVal = b.create<arith::ConstantIndexOp>(op.getPadStart());
      Value padEndVal = b.create<arith::ConstantIndexOp>(op.getPadEnd());

      emitKernelLaunch(b, op.getLoc(), kernelFunc, gridSize, blockDim,
                       ValueRange{outDyn, nttDyn, invPProdDyn, src2Dyn,
                                  src2PadDyn, padStartVal, padEndVal});
    }

    rewriter.eraseOp(op);
    return success();
  }
};

}  // namespace

void populateModDownEpilogueToGPUPatterns(RewritePatternSet& patterns) {
  patterns.add<ConvertModDownEpilogueOp>(patterns.getContext());
}

void configureModDownEpilogueGPUConversionTarget(ConversionTarget& target) {
  target.addIllegalOp<ModDownEpilogueOp>();
}

}  // namespace cheddar_gpu
}  // namespace heir
}  // namespace mlir
