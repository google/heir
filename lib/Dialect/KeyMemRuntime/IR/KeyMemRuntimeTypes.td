#ifndef LIB_DIALECT_KEYMEMRUNTIME_IR_KEYMEMRUNTIMETYPES_TD_
#define LIB_DIALECT_KEYMEMRUNTIME_IR_KEYMEMRUNTIMETYPES_TD_

include "lib/Dialect/KeyMemRuntime/IR/KeyMemRuntimeDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

class KeyMemRuntime_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<KeyMemRuntime_Dialect, name, traits # [OpAsmTypeInterface]> {
  let mnemonic = typeMnemonic;
  let assemblyFormat = "`<` struct(params) `>`";

  string asmName = ?;
  let extraClassDeclaration = [{
    // OpAsmTypeInterface method
    void getAsmName(::mlir::OpAsmSetNameFn setNameFn) const {
      setNameFn("}] # asmName # [{");
    }

    ::mlir::OpAsmDialectInterface::AliasResult getAlias(::llvm::raw_ostream &os) const {
      os << "}] # asmName # [{";
      return ::mlir::OpAsmDialectInterface::AliasResult::FinalAlias;
    }
  }];
}

def KeyMemRuntime_RotKeyType : KeyMemRuntime_Type<"RotKey", "rot_key", [MemRefElementTypeInterface]> {
  let summary = "A rotation key type with optional static index parameter";
  let description = [{
    The RotKeyType represents rotation keys used in the KeyMemRuntime system.

    - If rotation_index is present: Static constant rotation key
    - If rotation_index is absent: Dynamic rotation key (determined at runtime,
      e.g., from affine loop induction variables)

    For dynamic rotation keys, analysis passes examine the LoadKeyOp's index
    operand to determine the range of possible rotation values by tracing through
    SSA def-use chains and analyzing affine contexts.

    Examples:
      !kmrt.rot_key<rotation_index = 5>    // Static rotation key for index 5
      !kmrt.rot_key                        // Dynamic rotation key
  }];

  let parameters = (ins OptionalParameter<"std::optional<int64_t>", "std::nullopt">:$rotation_index);
  let asmName = "rk";

  // extraClassDeclaration in derived type replaces base class, so we must include OpAsmTypeInterface methods
  let extraClassDeclaration = [{
    // OpAsmTypeInterface method
    void getAsmName(::mlir::OpAsmSetNameFn setNameFn) const {
      setNameFn("rk");
    }

    ::mlir::OpAsmDialectInterface::AliasResult getAlias(::llvm::raw_ostream &os) const {
      os << "rk";
      return ::mlir::OpAsmDialectInterface::AliasResult::FinalAlias;
    }

    // Check if this is a static constant rotation key
    bool isStatic() const {
      return getRotationIndex().has_value();
    }

    // Check if this is a dynamic rotation key
    bool isDynamic() const {
      return !getRotationIndex().has_value();
    }

    // Get the static rotation index (only valid if isStatic())
    int64_t getStaticIndex() const {
      assert(isStatic() && "Not a static rotation key");
      return *getRotationIndex();
    }
  }];
}

#endif // LIB_DIALECT_KEYMEMRUNTIME_IR_KEYMEMRUNTIMETYPES_TD_
