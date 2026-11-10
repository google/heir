#ifndef LIB_DIALECT_CHEDDAR_IR_CHEDDARATTRIBUTES_TD_
#define LIB_DIALECT_CHEDDAR_IR_CHEDDARATTRIBUTES_TD_

include "lib/Dialect/Cheddar/IR/CheddarDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

class Cheddar_Attribute<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Cheddar_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
  let assemblyFormat = "`<` struct(params) `>`";
}

def Cheddar_ParameterSetAttr
    : Cheddar_Attribute<"ParameterSet", "parameter_set"> {
  let summary = "CHEDDAR runtime parameter set";
  let description = [{
    Records the parameters needed to construct a `cheddar::Parameter<word>` (and
    optionally bootstrapping level budgets when bootstrapping is enabled).
  }];
  let parameters = (ins
    AttrParameter<"int", "Log2 of the polynomial ring degree.">:$logN,
    AttrParameter<"int", "Log2 of the base scaling factor at level 0.">:$logScale,
    AttrParameter<"DenseI64ArrayAttr", "Array of main RNS primes.">:$mainPrimes,
    AttrParameter<"DenseI64ArrayAttr", "Array of auxiliary (special) RNS primes for key switching.">:$auxPrimes,
    OptionalParameter<"DenseI64ArrayAttr", "Optional array of terminal RNS primes.">:$terminalPrimes,
    OptionalParameter<"DenseI32ArrayAttr",
      "Optional flattened array of `[numMain_0, numTerminal_0, numMain_1, "
      "numTerminal_1, ...]` pairs for each level from level 0 up to maxLevel.">:$levelConfig,
    OptionalParameter<"std::optional<int>",
      "Optional word bit width (32 or 64, defaults to 32 if all primes < 2^31 else 64).">:$wordBits,
    OptionalParameter<"std::optional<int>",
      "Optional default encryption level. Must be a valid level index into "
      "`levelConfig` when both are present; levels above it are reserved for "
      "bootstrapping.">:$defaultEncryptionLevel,
    OptionalParameter<"DenseI32ArrayAttr",
      "Optional 2-element array `[numMain, numTerminal]` for additional base "
      "primes at level 0.">:$additionalBase,
    OptionalParameter<"std::optional<int>", "Optional dense secret key Hamming weight.">:$denseHammingWeight,
    OptionalParameter<"std::optional<int>", "Optional sparse secret key Hamming weight.">:$sparseHammingWeight,
    DefaultValuedParameter<"bool", "false",
      "Whether bootstrapping is enabled (defaults to false). When true, a "
      "`cheddar::BootContext` must be constructed and `numCtsLevels` and "
      "`numStcLevels` are required.">:$boot,
    OptionalParameter<"std::optional<int>", "Optional CoeffToSlot level budget for bootstrapping.">:$numCtsLevels,
    OptionalParameter<"std::optional<int>", "Optional SlotToCoeff level budget for bootstrapping.">:$numStcLevels
  );
  let genVerifyDecl = 1;
}

#endif  // LIB_DIALECT_CHEDDAR_IR_CHEDDARATTRIBUTES_TD_
