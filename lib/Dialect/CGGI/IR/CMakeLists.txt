set(LLVM_TARGET_DEFINITIONS CGGIDialect.td)
mlir_tablegen(CGGIDialect.h.inc -gen-dialect-decls -dialect=cggi)
mlir_tablegen(CGGIDialect.cpp.inc -gen-dialect-defs -dialect=cggi)
add_public_tablegen_target(MLIRCGGIIncGen)

set(LLVM_TARGET_DEFINITIONS CGGIOps.td)
mlir_tablegen(CGGIOps.h.inc -gen-op-decls)
mlir_tablegen(CGGIOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRCGGIOpsIncGen)

set(LLVM_TARGET_DEFINITIONS CGGIAttributes.td)
mlir_tablegen(CGGIAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect="cggi")
mlir_tablegen(CGGIAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect="cggi")
add_public_tablegen_target(MLIRCGGIAttributesIncGen)
