#ifndef LIB_DIALECT_POULPY_IR_POULPYENUMS_TD_
#define LIB_DIALECT_POULPY_IR_POULPYENUMS_TD_

include "mlir/IR/EnumAttr.td"
include "lib/Dialect/Poulpy/IR/PoulpyDialect.td"

def Poulpy_BackendEnum : I32EnumAttr<"PoulpyBackend", "An enum attribute representing a module's backend-specific handle", [
  I32EnumAttrCase<"FFT64Ref", 0, "fft64_ref">, // f64 FFT backend
  I32EnumAttrCase<"NTT4x30Ref", 1, "ntt4x30_ref"> // Q120 NTT backend (CRT over four ~30-bit primes)
]> {
    let cppNamespace = "::mlir::heir::poulpy";
}

#endif // LIB_DIALECT_POULPY_IR_POULPYENUMS_TD_
