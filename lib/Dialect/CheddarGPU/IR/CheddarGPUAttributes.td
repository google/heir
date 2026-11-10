#ifndef LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPU_ATTRIBUTES_TD_
#define LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPU_ATTRIBUTES_TD_

include "lib/Dialect/CheddarGPU/IR/CheddarGPUDialect.td"
include "lib/Dialect/CheddarGPU/IR/CheddarGPUEnums.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

class CheddarGPU_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<CheddarGPU_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

// === Form Enum Attribute ===

def CheddarGPU_FormAttr : EnumAttr<CheddarGPU_Dialect, CheddarGPU_FormEnum, "form">;

// === INTT Epilogue Enum Attribute ===

def CheddarGPU_INTTEpilogueAttr : EnumAttr<CheddarGPU_Dialect, CheddarGPU_INTTEpilogueEnum, "intt_epilogue">;

// === Params Attribute ===

def CheddarGPU_ParamsAttr : CheddarGPU_Attr<"Params", "params"> {
  let summary = "Cheddar GPU scheme parameters";
  let description = [{
    Module-level attribute encoding scheme parameters and modulus chain info.
  }];

  let parameters = (ins
    AttrParameter<"uint32_t", "word size in bits (32 or 64)">:$word,
    AttrParameter<"uint32_t", "log2 of the polynomial degree N = 2^log_n">:$log_n,
    AttrParameter<"::mlir::DenseI64ArrayAttr", "moduli for terminal primes (ter[0 ... num_ter - 1])">:$ter,
    AttrParameter<"::mlir::DenseI64ArrayAttr", "moduli for main primes (main[0 ... num_main - 1])">:$main,
    AttrParameter<"::mlir::DenseI64ArrayAttr", "moduli for auxiliary primes (aux[0 ... num_aux - 1])">:$aux,
    AttrParameter<"::mlir::ArrayAttr", "list of [m, t] pairs specifying level modulus configuration">:$level_config,
    AttrParameter<"uint32_t", "number of auxiliary primes per digit in key-switching">:$alpha,
    AttrParameter<"uint32_t", "number of digits in key-switching">:$dnum
  );

  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Total number of limbs across ter, main, and aux primes.
    size_t getNumLimbs() const;

    /// Number of terminal, main, and auxiliary primes.
    size_t getNumTer() const { return getTer().size(); }
    size_t getNumMain() const { return getMain().size(); }
    size_t getNumAux() const { return getAux().size(); }

    /// Returns prime at absolute limb index in Cheddar limb order:
    /// ter[num_ter - 1 ... 0], main[0 ... num_main - 1], aux[0 ... num_aux - 1].
    uint64_t getPrimeAt(size_t limbIdx) const;

    /// Returns absolute limb index of main prime `i`.
    size_t getLimbIndexOfMain(size_t i) const { return getTer().size() + i; }

    /// Returns absolute limb index of aux prime `i`.
    size_t getLimbIndexOfAux(size_t i) const { return getTer().size() + getMain().size() + i; }

    /// Returns absolute limb index of ter prime `i` (given in ter_primes array order).
    size_t getLimbIndexOfTer(size_t i) const { return getTer().size() - 1 - i; }
  }];
}

#endif  // LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPU_ATTRIBUTES_TD_
