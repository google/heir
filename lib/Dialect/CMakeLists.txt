add_subdirectory(ModArith)
# IR + Transforms
add_subdirectory(LWE)


add_subdirectory(Jaxite)
add_subdirectory(Openfhe)
add_subdirectory(RNS)
add_subdirectory(TfheRust)
add_subdirectory(TfheRustBool)
add_subdirectory(Polynomial)

add_subdirectory(BGV)
add_subdirectory(CGGI)
add_subdirectory(Secret)
add_subdirectory(Random)
add_subdirectory(CKKS)

set(LLVM_TARGET_DEFINITIONS HEIRInterfaces.td)
mlir_tablegen(HEIRInterfaces.h.inc  --gen-op-interface-decls  -name heir)
mlir_tablegen(HEIRInterfaces.cpp.inc  --gen-op-interface-defs -name heir)
add_public_tablegen_target(MLIRHEIRInterfacesOpsIncGen)

add_mlir_dialect_library(MLIRHEIRInterfaces
    HEIRInterfaces.cpp

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/IR

    DEPENDS
    MLIRHEIRInterfacesOpsIncGen
    HEIRIncludeMLIR
    HEIRIncludeLLVM

    LINK_LIBS PUBLIC
    LLVMSupport
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRSupport
    MLIRDialect
)


add_subdirectory(Comb) #- requires HEIR interface
add_dependencies(MLIRCombOpsIncGen MLIRHEIRInterfacesOpsIncGen)

add_subdirectory(TensorExt)
