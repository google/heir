#include "lib/Dialect/CheddarGPU/Conversions/CheddarGPUToGPU/KernelBuilders.h"

#include <cstddef>
#include <cstdint>
#include <optional>

#include "lib/Dialect/CheddarGPU/IR/CheddarGPUAttributes.h"
#include "lib/Dialect/CheddarGPU/Utils/ConstantTables.h"
#include "lib/Utils/AttributeUtils.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"               // from @llvm-project
#include "llvm/include/llvm/ADT/STLFunctionalExtras.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"              // from @llvm-project
#include "mlir/include/mlir/Dialect/DLTI/DLTI.h"          // from @llvm-project
#include "mlir/include/mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/GPU/IR/GPUDialectDecl.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"               // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Interfaces/DataLayoutInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar_gpu {

std::optional<ModuleParams> getModuleParams(Operation* op) {
  auto params =
      getAttrFromEnclosingModule<ParamsAttr>(op, "cheddar_gpu.params");
  if (!params) return std::nullopt;
  uint32_t logN = params.getLogN();
  if (logN >= 63) return std::nullopt;
  return ModuleParams{params, params.getWord(), logN, 1LL << logN};
}

Value castToDynamicFirstDim(OpBuilder& b, Location loc, Value val) {
  auto type = dyn_cast<MemRefType>(val.getType());
  if (!type || type.getRank() == 0 || type.isDynamicDim(0)) {
    return val;
  }
  SmallVector<int64_t> shape(type.getShape());
  shape[0] = ShapedType::kDynamic;
  auto dynamicType = MemRefType::get(shape, type.getElementType(),
                                     type.getLayout(), type.getMemorySpace());
  return memref::CastOp::create(b, loc, dynamicType, val);
}

void attachNvptxDataLayout(gpu::GPUModuleOp gpuMod) {
  if (gpuMod->hasAttr(DLTIDialect::kDataLayoutAttrName)) return;
  MLIRContext* ctx = gpuMod.getContext();
  auto i64Type = IntegerType::get(ctx, 64);
  auto i128Type = IntegerType::get(ctx, 128);
  auto i64Vec2 = VectorType::get({2}, i64Type);
  SmallVector<DataLayoutEntryInterface> entries = {
      DataLayoutEntryAttr::get(
          StringAttr::get(ctx, DLTIDialect::kDataLayoutEndiannessKey),
          StringAttr::get(ctx, DLTIDialect::kDataLayoutEndiannessLittle)),
      DataLayoutEntryAttr::get(
          i64Type,
          DenseIntElementsAttr::get(i64Vec2, ArrayRef<int64_t>{64, 64})),
      DataLayoutEntryAttr::get(
          i128Type,
          DenseIntElementsAttr::get(i64Vec2, ArrayRef<int64_t>{128, 128})),
      DataLayoutEntryAttr::get(
          StringAttr::get(ctx, DLTIDialect::kDataLayoutLegalIntWidthsKey),
          DenseI32ArrayAttr::get(ctx, ArrayRef<int32_t>{16, 32, 64})),
      DataLayoutEntryAttr::get(IndexType::get(ctx),
                               IntegerAttr::get(i64Type, 64)),
  };
  gpuMod->setAttr(DLTIDialect::kDataLayoutAttrName,
                  DataLayoutSpecAttr::get(ctx, entries));
}

gpu::GPUModuleOp getOrCreateGpuModule(ModuleOp module, StringRef moduleName) {
  if (auto gpuMod = module.lookupSymbol<gpu::GPUModuleOp>(moduleName)) {
    attachNvptxDataLayout(gpuMod);
    return gpuMod;
  }
  OpBuilder b(&module.getBodyRegion().front(),
              module.getBodyRegion().front().begin());
  auto gpuMod = b.create<gpu::GPUModuleOp>(module.getLoc(), moduleName);
  module->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                  UnitAttr::get(module.getContext()));
  attachNvptxDataLayout(gpuMod);
  return gpuMod;
}

memref::GlobalOp getOrCreateDevicePrimesGlobal(gpu::GPUModuleOp gpuMod,
                                               const ParamsAttr& params,
                                               uint32_t word) {
  auto primesGlobal = gpuMod.lookupSymbol<memref::GlobalOp>("__cheddar_primes");
  if (primesGlobal) {
    return primesGlobal;
  }

  OpBuilder b(gpuMod.getBody(), gpuMod.getBody()->begin());
  Location loc = gpuMod.getLoc();
  Type intType = IntegerType::get(gpuMod.getContext(), word);
  size_t totalLimbs = params.getNumLimbs();

  DenseElementsAttr primesDense =
      getDevicePrimesConstant(gpuMod.getContext(), params, word);
  auto memrefType =
      MemRefType::get({static_cast<int64_t>(totalLimbs)}, intType);

  return memref::GlobalOp::create(
      b, loc, "__cheddar_primes", b.getStringAttr("private"), memrefType,
      primesDense, /*constant=*/true, /*alignment=*/IntegerAttr());
}

Value getDevicePrimesGlobal(ImplicitLocOpBuilder& b, uint32_t word,
                            size_t totalLimbs) {
  MemRefType memrefType;
  if (auto gpuMod = b.getInsertionBlock()
                        ->getParentOp()
                        ->getParentOfType<gpu::GPUModuleOp>()) {
    if (auto g = gpuMod.lookupSymbol<memref::GlobalOp>("__cheddar_primes")) {
      memrefType = g.getType();
    }
  }
  if (!memrefType) {
    Type intType = IntegerType::get(b.getContext(), word);
    memrefType = MemRefType::get({static_cast<int64_t>(totalLimbs)}, intType);
  }
  return memref::GetGlobalOp::create(b, memrefType, "__cheddar_primes");
}

gpu::GPUFuncOp getOrCreateKernel(
    gpu::GPUModuleOp gpuMod, StringRef kernelName, FunctionType kernelType,
    llvm::function_ref<void(OpBuilder& b, Location loc, Block* entryBlock)>
        bodyBuilder) {
  if (auto existing = gpuMod.lookupSymbol<gpu::GPUFuncOp>(kernelName)) {
    return existing;
  }
  OpBuilder b(gpuMod.getBody(), gpuMod.getBody()->end());
  Location loc = gpuMod.getLoc();
  auto kernelFunc = b.create<gpu::GPUFuncOp>(loc, kernelName, kernelType);
  kernelFunc.setKernel(true);

  Block* entryBlock = kernelFunc.getBody().empty()
                          ? kernelFunc.addEntryBlock()
                          : &kernelFunc.getBody().front();
  OpBuilder bodyB(entryBlock, entryBlock->begin());
  bodyBuilder(bodyB, loc, entryBlock);

  if (entryBlock->empty() ||
      !entryBlock->back().hasTrait<OpTrait::IsTerminator>()) {
    OpBuilder termB(entryBlock, entryBlock->end());
    termB.create<gpu::ReturnOp>(loc);
  }
  return kernelFunc;
}

}  // namespace cheddar_gpu
}  // namespace heir
}  // namespace mlir
