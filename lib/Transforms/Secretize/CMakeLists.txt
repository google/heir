# Since this defines two passes in one *.td file,
# we cannot use the standard helper function.
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Secretize)
add_public_tablegen_target(HEIRSecretizeIncGen)

add_mlir_library(HEIRSecretize
    Secretize.cpp
    WrapGeneric.cpp

    DEPENDS
    HEIRSecretizeIncGen

    LINK_LIBS PUBLIC
    HEIRSecret
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRArithDialect
    MLIRSupport
    MLIRDialect
)
target_link_libraries(HEIRTransforms INTERFACE HEIRSecretize)
