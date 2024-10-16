include_guard()

# Make MLIR/LLVM CMake helper functions available
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddMLIR)
include(AddLLVM)
include(TableGen)

# custom TableGen helper for HEIR Dialects that follow our naming conventions
function(add_heir_dialect dialect dialect_namespace)
  add_custom_target(HEIR${dialect}IncGen)

  set(LLVM_TARGET_DEFINITIONS ${dialect}Dialect.td)
  mlir_tablegen(${dialect}Dialect.h.inc -gen-dialect-decls -dialect=${dialect_namespace})
  mlir_tablegen(${dialect}Dialect.cpp.inc -gen-dialect-defs -dialect=${dialect_namespace})
  add_public_tablegen_target(HEIR${dialect}DialectIncGen)
  add_dependencies(HEIR${dialect}IncGen HEIR${dialect}DialectIncGen)

  # Ops
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dialect}Ops.td")
    set(LLVM_TARGET_DEFINITIONS ${dialect}Ops.td)
    mlir_tablegen(${dialect}Ops.h.inc -gen-op-decls -dialect=${dialect_namespace})
    mlir_tablegen(${dialect}Ops.cpp.inc -gen-op-defs -dialect=${dialect_namespace})
    add_public_tablegen_target(HEIR${dialect}OpsIncGen)
    add_dependencies(HEIR${dialect}IncGen HEIR${dialect}OpsIncGen)
  endif()

  # Types
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dialect}Types.td")
    set(LLVM_TARGET_DEFINITIONS ${dialect}Types.td)
    mlir_tablegen(${dialect}Types.h.inc -gen-typedef-decls -dialect=${dialect_namespace})
    mlir_tablegen(${dialect}Types.cpp.inc -gen-typedef-defs -dialect=${dialect_namespace})
    add_public_tablegen_target(HEIR${dialect}TypesIncGen)
    add_dependencies(HEIR${dialect}IncGen HEIR${dialect}TypesIncGen)
  endif()

  # Attributes
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dialect}Attributes.td")
    set(LLVM_TARGET_DEFINITIONS ${dialect}Attributes.td)
    mlir_tablegen(${dialect}Attributes.h.inc -gen-attrdef-decls -attrdefs-dialect=${dialect_namespace})
    mlir_tablegen(${dialect}Attributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=${dialect_namespace})
    add_public_tablegen_target(HEIR${dialect}AttributesIncGen)
    add_dependencies(HEIR${dialect}IncGen HEIR${dialect}AttributesIncGen)
  endif()

  #Enums
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dialect}Enums.td")
    set(LLVM_TARGET_DEFINITIONS ${dialect}Enums.td)
    mlir_tablegen(${dialect}Enums.h.inc -gen-enum-decls)
    mlir_tablegen(${dialect}Enums.cpp.inc -gen-enum-defs)
    add_public_tablegen_target(HEIR${dialect}EnumsIncGen)
    add_dependencies(HEIR${dialect}IncGen HEIR${dialect}EnumsIncGen)
  #Enums (but from Attributes file)
  elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dialect}Attributes.td")
    set(LLVM_TARGET_DEFINITIONS ${dialect}Attributes.td)
    mlir_tablegen(${dialect}Enums.h.inc -gen-enum-decls)
    mlir_tablegen(${dialect}Enums.cpp.inc -gen-enum-defs)
    add_public_tablegen_target(HEIR${dialect}EnumsIncGen)
    add_dependencies(HEIR${dialect}IncGen HEIR${dialect}EnumsIncGen)
  endif()

  # Type Interfaces
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dialect}TypeInterfaces.td")
    set(LLVM_TARGET_DEFINITIONS ${dialect}TypeInterfaces.td)
    mlir_tablegen(${dialect}TypeInterfaces.h.inc -gen-type-interface-decls -name ${dialect_namespace})
    mlir_tablegen(${dialect}TypeInterfaces.cpp.inc -gen-type-interface-defs -name ${dialect_namespace})
    add_public_tablegen_target(HEIR${dialect}TypeInterfacesIncGen)
    add_dependencies(HEIR${dialect}IncGen HEIR${dialect}TypeInterfacesIncGen)
  endif()

endfunction() # add_heir_dialect


# custom TableGen helper for HEIR passes
# call add_heir_pass(someName) for normal use
# call add_heir_pass(someName PATTERNS) to generate rewriter patterns
function(add_heir_pass pass)
  set(LLVM_TARGET_DEFINITIONS ${pass}.td)
  mlir_tablegen(${pass}.h.inc -gen-pass-decls -name ${pass})

  if(ARGN MATCHES "PATTERNS")
    mlir_tablegen(${pass}.cpp.inc -gen-rewriters)
  endif()

  add_public_tablegen_target(HEIR${pass}IncGen)
endfunction() # add_heir_pass


## TODO: INTEGRATE THIS
# target_compile_features(my_lib PUBLIC cxx_std_17)
# if(CMAKE_CXX_STANDARD LESS 17)
#   message(FATAL_ERROR
#       "my_lib_project requires CMAKE_CXX_STANDARD >= 17 (got: ${CMAKE_CXX_STANDARD})")
# endif()
