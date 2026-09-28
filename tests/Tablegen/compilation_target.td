// RUN: heir-tblgen -gen-compilation-target-registration %s | FileCheck %s --check-prefix=REG
// RUN: heir-tblgen -gen-compilation-target-overrides %s | FileCheck %s --check-prefix=OVERRIDE
// RUN: heir-tblgen -gen-compilation-target-struct %s | FileCheck %s --check-prefix=STRUCT

class CompilationTarget<string name> {
  string backendName = name;
  int bootstrapLevelsConsumed = 0;
  int has_kernel_chebyshev = 0;
  int has_kernel_linear_transform = 0;
  int requires_matching_ciphertext_plaintext_levels = 0;
  int can_emit_adjust_scale = 1;
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
// OVERRIDE:   if (key == "has_kernel_chebyshev") {
// OVERRIDE:     auto attr = dyn_cast<IntegerAttr>(value);
// OVERRIDE:     if (!attr) { return emitError(loc, "expected integer attribute for key: ") << key; }
// OVERRIDE:     target.has_kernel_chebyshev = attr.getInt();
// OVERRIDE:     return success();
// OVERRIDE:   }
// OVERRIDE:   if (key == "has_kernel_linear_transform") {
// OVERRIDE:     auto attr = dyn_cast<IntegerAttr>(value);
// OVERRIDE:     if (!attr) { return emitError(loc, "expected integer attribute for key: ") << key; }
// OVERRIDE:     target.has_kernel_linear_transform = attr.getInt();
// OVERRIDE:     return success();
// OVERRIDE:   }
// OVERRIDE:   if (key == "requires_matching_ciphertext_plaintext_levels") {
// OVERRIDE:     auto attr = dyn_cast<IntegerAttr>(value);
// OVERRIDE:     if (!attr) { return emitError(loc, "expected integer attribute for key: ") << key; }
// OVERRIDE:     target.requires_matching_ciphertext_plaintext_levels = attr.getInt();
// OVERRIDE:     return success();
// OVERRIDE:   }
// OVERRIDE:   if (key == "can_emit_adjust_scale") {
// OVERRIDE:     auto attr = dyn_cast<IntegerAttr>(value);
// OVERRIDE:     if (!attr) { return emitError(loc, "expected integer attribute for key: ") << key; }
// OVERRIDE:     target.can_emit_adjust_scale = attr.getInt();
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
// OVERRIDE:   if (key == "has_kernel_chebyshev") {
// OVERRIDE:     int64_t intVal;
// OVERRIDE:     if (value.getAsInteger(10, intVal)) {
// OVERRIDE:       return emitError(loc, "expected base-10 integer attribute for key: ") << key;
// OVERRIDE:     }
// OVERRIDE:     return Attribute(IntegerAttr::get(IntegerType::get(ctx, 64), intVal));
// OVERRIDE:   }
// OVERRIDE:   if (key == "has_kernel_linear_transform") {
// OVERRIDE:     int64_t intVal;
// OVERRIDE:     if (value.getAsInteger(10, intVal)) {
// OVERRIDE:       return emitError(loc, "expected base-10 integer attribute for key: ") << key;
// OVERRIDE:     }
// OVERRIDE:     return Attribute(IntegerAttr::get(IntegerType::get(ctx, 64), intVal));
// OVERRIDE:   }
// OVERRIDE:   if (key == "requires_matching_ciphertext_plaintext_levels") {
// OVERRIDE:     int64_t intVal;
// OVERRIDE:     if (value.getAsInteger(10, intVal)) {
// OVERRIDE:       return emitError(loc, "expected base-10 integer attribute for key: ") << key;
// OVERRIDE:     }
// OVERRIDE:     return Attribute(IntegerAttr::get(IntegerType::get(ctx, 64), intVal));
// OVERRIDE:   }
// OVERRIDE:   if (key == "can_emit_adjust_scale") {
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
// STRUCT:   int64_t has_kernel_chebyshev = 0;
// STRUCT:   int64_t has_kernel_linear_transform = 0;
// STRUCT:   int64_t requires_matching_ciphertext_plaintext_levels = 0;
// STRUCT:   int64_t can_emit_adjust_scale = 1;
// STRUCT: };
