set(LLVM_TARGET_DEFINITIONS PolyExtDialect.td)
mlir_tablegen(PolyExtDialect.h.inc -gen-dialect-decls)# -dialect=polyext)
mlir_tablegen(PolyExtDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRPolyExtIncGen)

set(LLVM_TARGET_DEFINITIONS PolyExtOps.td)
mlir_tablegen(PolyExtOps.h.inc -gen-op-decls -dialect=polyext)
mlir_tablegen(PolyExtOps.cpp.inc -gen-op-defs -dialect=polyext)
add_public_tablegen_target(MLIRPolyExtOpsIncGen)
