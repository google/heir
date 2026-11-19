#include "lib/Dialect/CheddarGPU/Conversions/CheddarGPUToGPU/CheddarGPUToGPU.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "lib/Dialect/CheddarGPU/Conversions/CheddarGPUToGPU/KernelBuilders.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUAttributes.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUOps.h"
#include "llvm/include/llvm/ADT/STLFunctionalExtras.h"  // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"   // from @llvm-project
#include "llvm/include/llvm/Support/MathExtras.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/ArithAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/ArithDialect.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/GPU/IR/GPUDialectDecl.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRefDialect.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Interfaces/DestinationStyleOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar_gpu {

#define GEN_PASS_DEF_CHEDDARGPUTOGPU
#include "lib/Dialect/CheddarGPU/Conversions/CheddarGPUToGPU/CheddarGPUToGPU.h.inc"

namespace {

constexpr int64_t kBlockDim = 256;

// Generic binary elementwise helper
template <typename OpTy>
static LogicalResult rewriteBinaryElementwiseOp(
    OpTy op, PatternRewriter& rewriter, StringRef opName,
    llvm::function_ref<Value(ImplicitLocOpBuilder& b, Value lhs, Value rhs,
                             Value q)>
        computeFn) {
  if (auto dstOp = dyn_cast<DestinationStyleOpInterface>(op.getOperation())) {
    if (!dstOp.hasPureBufferSemantics()) {
      return op.emitOpError(
          "requires buffer operands; run --one-shot-bufferize before "
          "--cheddar-gpu-to-gpu");
    }
  }

  auto modParams = getModuleParams(op);
  if (!modParams) return op.emitOpError("missing cheddar_gpu.params on module");

  auto outType = cast<MemRefType>(op.getOut().getType());
  int64_t outCount = outType.getDimSize(0);
  int64_t degree = modParams->degree;
  uint32_t word = modParams->word;
  size_t totalLimbs = modParams->params.getNumLimbs();

  // Launch geometry: One thread processes one (limb, coeff) element.
  // Using 1-D block dimension kBlockDim=256, the 1-D grid size is
  // (outCount * degree) / 256. Enforce that total threads is positive
  // and evenly divisible by kBlockDim.
  int64_t totalThreads = outCount * degree;
  if (totalThreads <= 0 || totalThreads % kBlockDim != 0) {
    return op.emitOpError(
        llvm::formatv("total threads ({0}) must be positive and divisible by "
                      "block dimension ({1})",
                      totalThreads, kBlockDim));
  }

  // Overflow guard: thread indexing and limb offset arithmetic within the
  // kernel are computed using 32-bit signed integers. Verify that degree and
  // total threads do not exceed INT32_MAX.
  constexpr int64_t kMaxI32 = std::numeric_limits<int32_t>::max();
  if (degree > kMaxI32 || totalThreads > kMaxI32) {
    return op.emitOpError(
        "elements or total threads exceed 32-bit signed integer range");
  }

  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  auto module = op->template getParentOfType<ModuleOp>();
  gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);

  // Memoize the kernel function. The arguments are memrefs storing the
  // ciphertext values, but with a dynamically-sized first axis so that the same
  // kernel can be used with ciphertexts having any number of limbs.
  // The arguments are (out, lhs, rhs, primeOffset, numQ, auxStart)
  std::string kernelName =
      llvm::formatv("cheddar_gpu_{0}_w{1}_deg{2}", opName, word, degree);

  Type elemType = outType.getElementType();
  auto dynMemrefType =
      MemRefType::get({ShapedType::kDynamic, degree}, elemType);
  Type i32Type = b.getI32Type();
  auto kernelType = FunctionType::get(
      b.getContext(),
      {dynMemrefType, dynMemrefType, dynMemrefType, i32Type, i32Type, i32Type},
      {});

  gpu::GPUFuncOp kernelFunc = getOrCreateKernel(
      gpuMod, kernelName, kernelType,
      [&](OpBuilder& kb, Location loc, Block* entryBlock) {
        ImplicitLocOpBuilder ib(loc, kb);
        Value outArg = entryBlock->getArgument(0);
        Value lhsArg = entryBlock->getArgument(1);
        Value rhsArg = entryBlock->getArgument(2);
        // These tell the kernel which primes to load from the global table of
        // primeOffset is the starting index in the table for limb 0,
        // numQ is the number of main primes (rest are auxiliary primes), and
        // auxStart is the starting index of auxiliary primes in the table.
        Value primeOffsetArg = entryBlock->getArgument(3);
        Value numQArg = entryBlock->getArgument(4);
        Value auxStartArg = entryBlock->getArgument(5);

        // Compute the thread ID from the block and thread index
        //
        // tid = blockIdx.x * blockDim.x + threadIdx.x
        Value bxIndex = gpu::BlockIdOp::create(ib, gpu::Dimension::x);
        Value txIndex = gpu::ThreadIdOp::create(ib, gpu::Dimension::x);
        Value bdimIndex = gpu::BlockDimOp::create(ib, gpu::Dimension::x);
        Value bx = arith::IndexCastOp::create(ib, i32Type, bxIndex);
        Value tx = arith::IndexCastOp::create(ib, i32Type, txIndex);
        Value bdim = arith::IndexCastOp::create(ib, i32Type, bdimIndex);
        Value bxDim = arith::MulIOp::create(ib, bx, bdim);
        Value tid = arith::AddIOp::create(ib, bxDim, tx);

        // Compute the "local" limb index, i.e., the index of the limb
        // being computed by this thread, and the coefficient being computed.
        //
        // localLimb = tid >> log2(degree) (degree is a power of 2)
        // coeff = tid & (degree - 1)
        assert(llvm::isPowerOf2_64(degree) && "degree must be power of 2");
        int64_t log2Degree = llvm::Log2_64(degree);
        Value cLog2Degree = arith::ConstantIntOp::create(ib, log2Degree, 32);
        Value cDegreeM1 = arith::ConstantIntOp::create(ib, degree - 1, 32);
        Value localLimb = arith::ShRUIOp::create(ib, tid, cLog2Degree);
        Value coeff = arith::AndIOp::create(ib, tid, cDegreeM1);

        // Each thread computes a single coefficient on a single limb,
        // which means it loads a single prime from the prime table for
        // its modulus op.
        //
        // if limbs < numQ: prime_offset + localLimb
        // else use auxStart + (localLimb - numQ).
        Value isAux = arith::CmpIOp::create(ib, arith::CmpIPredicate::sge,
                                            localLimb, numQArg);
        Value qLimbAbsIndex =
            arith::AddIOp::create(ib, localLimb, primeOffsetArg);
        Value auxOffset = arith::SubIOp::create(ib, localLimb, numQArg);
        Value auxLimbAbsIndex =
            arith::AddIOp::create(ib, auxOffset, auxStartArg);
        Value limbAbsIndex =
            arith::SelectOp::create(ib, isAux, auxLimbAbsIndex, qLimbAbsIndex);

        // Load the chosen prime
        Type indexType = ib.getIndexType();
        Value primeIndex =
            arith::IndexCastOp::create(ib, indexType, limbAbsIndex);
        Value primesMemref = getDevicePrimesGlobal(ib, word, totalLimbs);
        Value q =
            memref::LoadOp::create(ib, primesMemref, ValueRange{primeIndex});

        // Load the chosen limb and coefficient
        Value coeffIndex = arith::IndexCastOp::create(ib, indexType, coeff);
        Value localLimbIndex =
            arith::IndexCastOp::create(ib, indexType, localLimb);

        Value lhsVal = memref::LoadOp::create(
            ib, lhsArg, ValueRange{localLimbIndex, coeffIndex});
        Value rhsVal = memref::LoadOp::create(
            ib, rhsArg, ValueRange{localLimbIndex, coeffIndex});

        // Run the computation and store the result.
        Value res = computeFn(ib, lhsVal, rhsVal, q);
        memref::StoreOp::create(ib, res, outArg,
                                ValueRange{localLimbIndex, coeffIndex});
        gpu::ReturnOp::create(ib);
      });

  int64_t numAux = 0;
  if (auto numAuxAttr = op.getNumAux()) numAux = *numAuxAttr;
  int64_t alpha = modParams->params.getAux().size();
  int64_t numQ = outCount - numAux;
  int64_t auxStart = totalLimbs - alpha;

  // Cast buffers to memref<?xdegree> so a single memoized GPU kernel per
  // (word, degree) can process operands of any limb count, with prime offset
  // and active counts passed as dynamic launch arguments.
  Value outDyn = castToDynamicFirstDim(b, op.getLoc(), op.getOut());
  Value lhsDyn = castToDynamicFirstDim(b, op.getLoc(), op.getLhs());
  Value rhsDyn = castToDynamicFirstDim(b, op.getLoc(), op.getRhs());
  Value primeOffsetVal = arith::ConstantIntOp::create(b, 0, 32);
  Value numQVal = arith::ConstantIntOp::create(b, numQ, 32);
  Value auxStartVal = arith::ConstantIntOp::create(b, auxStart, 32);

  int64_t gridDimX = totalThreads / kBlockDim;
  Value c1 = arith::ConstantIndexOp::create(b, 1);
  Value gridX = arith::ConstantIndexOp::create(b, gridDimX);
  Value blockX = arith::ConstantIndexOp::create(b, kBlockDim);

  gpu::LaunchFuncOp::create(
      b, op.getLoc(), kernelFunc, gpu::KernelDim3{gridX, c1, c1},
      gpu::KernelDim3{blockX, c1, c1}, /*dynamicSharedMemorySize=*/Value(),
      ValueRange{outDyn, lhsDyn, rhsDyn, primeOffsetVal, numQVal, auxStartVal});

  rewriter.eraseOp(op);
  return success();
}

// 1. AddOp: out = lhs + rhs mod q
struct ConvertAddOp : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter& rewriter) const override {
    return rewriteBinaryElementwiseOp<AddOp>(
        op, rewriter, "add",
        [](ImplicitLocOpBuilder& b, Value lhs, Value rhs, Value q) -> Value {
          Value sum = arith::AddIOp::create(b, lhs, rhs);
          Value isGe =
              arith::CmpIOp::create(b, arith::CmpIPredicate::uge, sum, q);
          Value sumSubQ = arith::SubIOp::create(b, sum, q);
          return arith::SelectOp::create(b, isGe, sumSubQ, sum);
        });
  }
};

// 2. SubOp: out = lhs - rhs mod q
struct ConvertSubOp : public OpRewritePattern<SubOp> {
  using OpRewritePattern<SubOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubOp op,
                                PatternRewriter& rewriter) const override {
    return rewriteBinaryElementwiseOp<SubOp>(
        op, rewriter, "sub",
        [](ImplicitLocOpBuilder& b, Value lhs, Value rhs, Value q) -> Value {
          Value diff = arith::SubIOp::create(b, lhs, rhs);
          Value isLt =
              arith::CmpIOp::create(b, arith::CmpIPredicate::ult, lhs, rhs);
          Value diffAddQ = arith::AddIOp::create(b, diff, q);
          return arith::SelectOp::create(b, isLt, diffAddQ, diff);
        });
  }
};

struct CheddarGPUToGPU : public impl::CheddarGPUToGPUBase<CheddarGPUToGPU> {
  using impl::CheddarGPUToGPUBase<CheddarGPUToGPU>::CheddarGPUToGPUBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    ModuleOp module = getOperation();

    auto paramsAttr = module->getAttrOfType<ParamsAttr>("cheddar_gpu.params");
    if (paramsAttr) {
      gpu::GPUModuleOp gpuMod = getOrCreateGpuModule(module);
      getOrCreateDevicePrimesGlobal(gpuMod, paramsAttr, paramsAttr.getWord());
    }

    RewritePatternSet patterns(context);
    patterns.add<ConvertAddOp, ConvertSubOp>(context);

    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, gpu::GPUDialect,
                           memref::MemRefDialect>();
    target.addIllegalOp<AddOp, SubOp>();

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createCheddarGPUToGPUPass() {
  return std::make_unique<CheddarGPUToGPU>();
}

}  // namespace cheddar_gpu
}  // namespace heir
}  // namespace mlir
