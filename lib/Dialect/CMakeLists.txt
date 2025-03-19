set(LLVM_TARGET_DEFINITIONS HEIRInterfaces.td)
mlir_tablegen(HEIRInterfaces.h.inc  --gen-op-interface-decls  -name heir)
mlir_tablegen(HEIRInterfaces.cpp.inc  --gen-op-interface-defs -name heir)
add_public_tablegen_target(HEIRInterfacesOpsIncGen)

add_mlir_dialect_library(HEIRInterfaces
    HEIRInterfaces.cpp

    DEPENDS
    HEIRInterfacesOpsIncGen
)

add_subdirectory(BGV)
add_subdirectory(CGGI)
add_subdirectory(CKKS)
add_subdirectory(Comb)
add_subdirectory(Jaxite)
add_subdirectory(JaxiteWord)
add_subdirectory(LinAlg)
add_subdirectory(LWE)
add_subdirectory(ModArith)
add_subdirectory(Openfhe)
add_subdirectory(Polynomial)
add_subdirectory(Random)
add_subdirectory(RNS)
add_subdirectory(Secret)
add_subdirectory(TensorExt)
add_subdirectory(TfheRust)
add_subdirectory(TfheRustBool)
add_subdirectory(TOSA)
