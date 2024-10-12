set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Openfhe)
add_public_tablegen_target(HEIROpenfhePassesIncGen)


add_mlir_library(HEIROpenfheTransforms
    ConfigureCryptoContext.cpp

    DEPENDS
    HEIROpenfhePassesIncGen

    LINK_LIBS PUBLIC
    HEIROpenfhe

    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransformUtils
  )
