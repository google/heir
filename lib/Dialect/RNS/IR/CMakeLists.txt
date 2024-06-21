set(LLVM_TARGET_DEFINITIONS RNSDialect.td)
mlir_tablegen(RNSDialect.h.inc -gen-dialect-decls -dialect=rns)
mlir_tablegen(RNSDialect.cpp.inc -gen-dialect-defs -dialect=rns)
add_public_tablegen_target(MLIRRNSIncGen)

set(LLVM_TARGET_DEFINITIONS RNSOps.td)
mlir_tablegen(RNSOps.h.inc -gen-op-decls)
mlir_tablegen(RNSOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRRNSOpsIncGen)

set(LLVM_TARGET_DEFINITIONS RNSTypes.td)
mlir_tablegen(RNSTypes.h.inc  --gen-typedef-decls  -name rns)
mlir_tablegen(RNSTypes.cpp.inc  --gen-typedef-defs -name rns)
add_public_tablegen_target(MLIRRNSTypeIncGen)

set(LLVM_TARGET_DEFINITIONS RNSTypeInterfaces.td)
mlir_tablegen(RNSTypeInterfaces.h.inc  --gen-type-interface-decls  -name rns)
mlir_tablegen(RNSTypeInterfaces.cpp.inc  --gen-type-interface-defs -name rns)
add_public_tablegen_target(MLIRRNSTypeInterfaces)
