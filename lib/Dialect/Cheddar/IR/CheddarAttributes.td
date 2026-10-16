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

    - `logN`: Log2 of the polynomial ring degree.
    - `logScale`: Log2 of the base scaling factor at level 0.
    - `mainPrimes`: Array of main RNS primes.
    - `auxPrimes`: Array of auxiliary (special) RNS primes for key switching.
    - `terminalPrimes`: Optional array of terminal RNS primes.
    - `levelConfig`: Optional flattened array of `[numMain_0, numTerminal_0, numMain_1, numTerminal_1, ...]` pairs for each level from level 0 up to maxLevel.
    - `wordBits`: Optional word bit width (32 or 64, defaults to 32 if all primes < 2^31 else 64).
    - `defaultEncryptionLevel`: Optional default encryption level.
    - `additionalBase`: Optional 2-element array `[numMain, numTerminal]` for additional base primes at level 0.
    - `denseHammingWeight`: Optional dense secret key Hamming weight.
    - `sparseHammingWeight`: Optional sparse secret key Hamming weight.
    - `numCtsLevels`: Optional CoeffToSlot level budget for bootstrapping.
    - `numStcLevels`: Optional SlotToCoeff level budget for bootstrapping.
  }];
  let parameters = (ins
    "int":$logN,
    "int":$logScale,
    "DenseI64ArrayAttr":$mainPrimes,
    "DenseI64ArrayAttr":$auxPrimes,
    OptionalParameter<"DenseI64ArrayAttr">:$terminalPrimes,
    OptionalParameter<"DenseI32ArrayAttr">:$levelConfig,
    OptionalParameter<"std::optional<int>">:$wordBits,
    OptionalParameter<"std::optional<int>">:$defaultEncryptionLevel,
    OptionalParameter<"DenseI32ArrayAttr">:$additionalBase,
    OptionalParameter<"std::optional<int>">:$denseHammingWeight,
    OptionalParameter<"std::optional<int>">:$sparseHammingWeight,
    OptionalParameter<"std::optional<int>">:$numCtsLevels,
    OptionalParameter<"std::optional<int>">:$numStcLevels
  );
  let genVerifyDecl = 1;
}

#endif  // LIB_DIALECT_CHEDDAR_IR_CHEDDARATTRIBUTES_TD_
