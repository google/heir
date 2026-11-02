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
    Fields:
    - `word`: word size in bits (32 or 64).
    - `log_n`: log2 of the polynomial degree N = 2^log_n.
    - `ter`: moduli for terminal primes (ter[0 ... num_ter - 1]).
    - `main`: moduli for main primes (main[0 ... num_main - 1]).
    - `aux`: moduli for auxiliary primes (aux[0 ... num_aux - 1]).
    - `level_config`: list of [m, t] pairs specifying level modulus configuration.
    - `alpha`: number of auxiliary primes per digit in key-switching.
    - `dnum`: number of digits in key-switching.
  }];

  let parameters = (ins
    "uint32_t":$word,
    "uint32_t":$log_n,
    "DenseI64ArrayAttr":$ter,
    "DenseI64ArrayAttr":$main,
    "DenseI64ArrayAttr":$aux,
    "::mlir::ArrayAttr":$level_config,
    "uint32_t":$alpha,
    "uint32_t":$dnum
  );

  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Total number of limbs across ter, main, and aux primes.
    size_t getNumLimbs() const;
    size_t numLimbs() const { return getNumLimbs(); }

    /// Number of terminal, main, and auxiliary primes.
    size_t getNumTer() const { return getTer().size(); }
    size_t numTer() const { return getNumTer(); }
    size_t getNumMain() const { return getMain().size(); }
    size_t numMain() const { return getNumMain(); }
    size_t getNumAux() const { return getAux().size(); }
    size_t numAux() const { return getNumAux(); }

    /// Returns prime at absolute limb index in Cheddar limb order:
    /// ter[num_ter - 1 ... 0], main[0 ... num_main - 1], aux[0 ... num_aux - 1].
    uint64_t getPrimeAt(size_t limbIdx) const;
    uint64_t primeAt(size_t limbIdx) const { return getPrimeAt(limbIdx); }

    /// Returns absolute limb index of main prime `i`.
    size_t getLimbIndexOfMain(size_t i) const { return getTer().size() + i; }
    size_t limbIndexOfMain(size_t i) const { return getLimbIndexOfMain(i); }

    /// Returns absolute limb index of aux prime `i`.
    size_t getLimbIndexOfAux(size_t i) const { return getTer().size() + getMain().size() + i; }
    size_t limbIndexOfAux(size_t i) const { return getLimbIndexOfAux(i); }

    /// Returns absolute limb index of ter prime `i` (given in ter_primes array order).
    size_t getLimbIndexOfTer(size_t i) const { return getTer().size() - 1 - i; }
    size_t limbIndexOfTer(size_t i) const { return getLimbIndexOfTer(i); }

    /// Factory method to construct ParamsAttr from a cheddar_ref::Params struct.
    template <unsigned W>
    static ParamsAttr fromParams(::mlir::MLIRContext *context,
                                 const ::mlir::heir::cheddar_ref::Params<W> &params);
  }];
}

#endif  // LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPU_ATTRIBUTES_TD_
