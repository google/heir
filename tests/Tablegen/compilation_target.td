// RUN: heir-tblgen -gen-compilation-target-registration %s | FileCheck %s --check-prefix=REG
// RUN: heir-tblgen -gen-compilation-target-overrides %s | FileCheck %s --check-prefix=OVERRIDE
// RUN: heir-tblgen -gen-compilation-target-struct %s | FileCheck %s --check-prefix=STRUCT

class CompilationTarget<string name> {
  string backendName = name;
  int bootstrapLevelsConsumed = 0;
  int hasKernelChebyshev = 0;
  int hasKernelLinearTransform = 0;
  int requiresMatchingCiphertextPlaintextLevels = 0;
  int supportsAdjustScale = 1;
}

def TestTarget : CompilationTarget<"test_target"> {
  let bootstrapLevelsConsumed = 1;
}

// REG: void registerTargetTestTarget() {
// REG:   CompilationTargetRegistry::registerTarget(CompilationTarget{BackendName::TestTarget, 1, 0, 0, 0, 1});
// REG: }

// OVERRIDE: mlir::LogicalResult applyCompilationTargetOverride(CompilationTarget& target,
// OVERRIDE:                                                     StringRef key,
// OVERRIDE:                                                     Attribute value,
// OVERRIDE:                                                     Location loc) {
// OVERRIDE:   if (key == "bootstrapLevelsConsumed") {
// OVERRIDE:     auto attr = dyn_cast<IntegerAttr>(value);
// OVERRIDE:     if (!attr) { return emitError(loc, "expected integer attribute for key: ") << key; }
// OVERRIDE:     target.bootstrapLevelsConsumed = attr.getInt();
// OVERRIDE:     return success();
// OVERRIDE:   }
// OVERRIDE:   if (key == "hasKernelChebyshev") {
// OVERRIDE:     auto attr = dyn_cast<IntegerAttr>(value);
// OVERRIDE:     if (!attr) { return emitError(loc, "expected integer attribute for key: ") << key; }
// OVERRIDE:     target.hasKernelChebyshev = attr.getInt();
// OVERRIDE:     return success();
// OVERRIDE:   }
// OVERRIDE:   if (key == "hasKernelLinearTransform") {
// OVERRIDE:     auto attr = dyn_cast<IntegerAttr>(value);
// OVERRIDE:     if (!attr) { return emitError(loc, "expected integer attribute for key: ") << key; }
// OVERRIDE:     target.hasKernelLinearTransform = attr.getInt();
// OVERRIDE:     return success();
// OVERRIDE:   }
// OVERRIDE:   if (key == "requiresMatchingCiphertextPlaintextLevels") {
// OVERRIDE:     auto attr = dyn_cast<IntegerAttr>(value);
// OVERRIDE:     if (!attr) { return emitError(loc, "expected integer attribute for key: ") << key; }
// OVERRIDE:     target.requiresMatchingCiphertextPlaintextLevels = attr.getInt();
// OVERRIDE:     return success();
// OVERRIDE:   }
// OVERRIDE:   if (key == "supportsAdjustScale") {
// OVERRIDE:     auto attr = dyn_cast<IntegerAttr>(value);
// OVERRIDE:     if (!attr) { return emitError(loc, "expected integer attribute for key: ") << key; }
// OVERRIDE:     target.supportsAdjustScale = attr.getInt();
// OVERRIDE:     return success();
// OVERRIDE:   }
// OVERRIDE:   return emitError(loc, "Cannot override backend config for unknown key: ") << key;
// OVERRIDE: }

// OVERRIDE: FailureOr<Attribute> parseCompilationTargetOverrideValue(MLIRContext* ctx,
// OVERRIDE:                                                          StringRef key,
// OVERRIDE:                                                          StringRef value,
// OVERRIDE:                                                          Location loc) {
// OVERRIDE:   if (key == "bootstrapLevelsConsumed") {
// OVERRIDE:     int64_t intVal;
// OVERRIDE:     if (value.getAsInteger(10, intVal)) {
// OVERRIDE:       return emitError(loc, "expected base-10 integer attribute for key: ") << key;
// OVERRIDE:     }
// OVERRIDE:     return Attribute(IntegerAttr::get(IntegerType::get(ctx, 64), intVal));
// OVERRIDE:   }
// OVERRIDE:   if (key == "hasKernelChebyshev") {
// OVERRIDE:     int64_t intVal;
// OVERRIDE:     if (value.getAsInteger(10, intVal)) {
// OVERRIDE:       return emitError(loc, "expected base-10 integer attribute for key: ") << key;
// OVERRIDE:     }
// OVERRIDE:     return Attribute(IntegerAttr::get(IntegerType::get(ctx, 64), intVal));
// OVERRIDE:   }
// OVERRIDE:   if (key == "hasKernelLinearTransform") {
// OVERRIDE:     int64_t intVal;
// OVERRIDE:     if (value.getAsInteger(10, intVal)) {
// OVERRIDE:       return emitError(loc, "expected base-10 integer attribute for key: ") << key;
// OVERRIDE:     }
// OVERRIDE:     return Attribute(IntegerAttr::get(IntegerType::get(ctx, 64), intVal));
// OVERRIDE:   }
// OVERRIDE:   if (key == "requiresMatchingCiphertextPlaintextLevels") {
// OVERRIDE:     int64_t intVal;
// OVERRIDE:     if (value.getAsInteger(10, intVal)) {
// OVERRIDE:       return emitError(loc, "expected base-10 integer attribute for key: ") << key;
// OVERRIDE:     }
// OVERRIDE:     return Attribute(IntegerAttr::get(IntegerType::get(ctx, 64), intVal));
// OVERRIDE:   }
// OVERRIDE:   if (key == "supportsAdjustScale") {
// OVERRIDE:     int64_t intVal;
// OVERRIDE:     if (value.getAsInteger(10, intVal)) {
// OVERRIDE:       return emitError(loc, "expected base-10 integer attribute for key: ") << key;
// OVERRIDE:     }
// OVERRIDE:     return Attribute(IntegerAttr::get(IntegerType::get(ctx, 64), intVal));
// OVERRIDE:   }
// OVERRIDE:   return emitError(loc, "Cannot parse backend config for unknown key: ") << key;
// OVERRIDE: }

// STRUCT: enum class BackendName {
// STRUCT:   None,
// STRUCT:   TestTarget,
// STRUCT: };

// STRUCT: inline llvm::StringRef toString(BackendName backend) {
// STRUCT:   switch (backend) {
// STRUCT:     case BackendName::None: return "none";
// STRUCT:     case BackendName::TestTarget: return "test_target";
// STRUCT:   }
// STRUCT: }

// STRUCT: inline BackendName parseBackendName(llvm::StringRef name) {
// STRUCT:   if (name == "test_target") return BackendName::TestTarget;
// STRUCT:   return BackendName::None;
// STRUCT: }

// STRUCT: struct CompilationTarget {
// STRUCT:   BackendName backendName = BackendName::None;
// STRUCT:   int64_t bootstrapLevelsConsumed = 0;
// STRUCT:   int64_t hasKernelChebyshev = 0;
// STRUCT:   int64_t hasKernelLinearTransform = 0;
// STRUCT:   int64_t requiresMatchingCiphertextPlaintextLevels = 0;
// STRUCT:   int64_t supportsAdjustScale = 1;
// STRUCT: };
