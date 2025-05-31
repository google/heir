set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name LWE)
add_public_tablegen_target(HEIRLWEPassesIncGen)


add_mlir_library(HEIRLWETransforms
    SetDefaultParameters.cpp

    DEPENDS
    HEIRLWEPassesIncGen

    LINK_LIBS PUBLIC
    HEIRLWE

    MLIRIR
    MLIRPass
    MLIRTransformUtils
  )
