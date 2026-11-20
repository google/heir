#ifndef LIB_DIALECT_CHEDDARGPU_CONVERSIONS_CHEDDARGPUTOGPU_KERNELBUILDERS_H_
#define LIB_DIALECT_CHEDDARGPU_CONVERSIONS_CHEDDARGPUTOGPU_KERNELBUILDERS_H_

#include <cstdint>
#include <optional>

#include "lib/Dialect/CheddarGPU/IR/CheddarGPUAttributes.h"
#include "llvm/include/llvm/ADT/StringRef.h"              // from @llvm-project
#include "mlir/include/mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"                // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                   // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"              // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar_gpu {

/// Module-level parameter struct extracted from module attribute.
struct ModuleParams {
  ParamsAttr params;
  uint32_t word;
  uint32_t logN;
  int64_t degree;
};

/// Name of the singleton GPU module that collects all CheddarGPU kernels.
inline constexpr StringLiteral kCheddarGpuModuleName = "cheddar_kernels";

/// Extracts module-level parameters from the module containing `op`.
std::optional<ModuleParams> getModuleParams(Operation* op);

/// Helper to cast the outermost dimension of a MemRef to dynamic.
Value castToDynamicFirstDim(OpBuilder& b, Location loc, Value val);

/// Attaches NVPTX64 data layout specification to the GPU module so that 64-bit
/// memory accesses are properly aligned and lowered to 64-bit PTX memory ops.
void attachNvptxDataLayout(gpu::GPUModuleOp gpuMod);

/// Finds or creates the singleton gpu.module named `kCheddarGpuModuleName`
/// in the given parent ModuleOp, and marks the parent module as a container
/// module.
gpu::GPUModuleOp getOrCreateGpuModule(
    ModuleOp module, StringRef moduleName = kCheddarGpuModuleName);

/// Finds or creates the global memref for primes inside gpuMod.
memref::GlobalOp getOrCreateDevicePrimesGlobal(gpu::GPUModuleOp gpuMod,
                                               const ParamsAttr& params,
                                               uint32_t word);

/// Helper to get Value handle to the device prime global inside a GPU kernel.
Value getDevicePrimesGlobal(ImplicitLocOpBuilder& b, uint32_t word,
                            size_t totalLimbs = 0);

/// Finds or creates a memoized gpu.func kernel in `gpuMod`.
gpu::GPUFuncOp getOrCreateKernel(
    gpu::GPUModuleOp gpuMod, StringRef kernelName, FunctionType kernelType,
    llvm::function_ref<void(OpBuilder& b, Location loc, Block* entryBlock)>
        bodyBuilder);

}  // namespace cheddar_gpu
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CHEDDARGPU_CONVERSIONS_CHEDDARGPUTOGPU_KERNELBUILDERS_H_
