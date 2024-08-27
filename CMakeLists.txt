cmake_minimum_required(VERSION 3.13.4)

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

project(HEIR LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

option(ENABLE_YOSYS "Enable YOSYS for yosys-optimizer pass (optional)" FALSE)
option(ENABLE_INSTALLED_MLIR "Use installed MLIR" FALSE)

if(ENABLE_YOSYS)
   message("Yosys enabled! Make sure ABC and Yosys development files and libraries are on the path")
else()
   add_compile_options(-DHEIR_NO_YOSYS=1)
endif()


set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})



list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

set(INCLUDE_MLIR_LLVM_HEADERS "${CMAKE_BINARY_DIR}/inc/")
add_custom_command(OUTPUT ${INCLUDE_MLIR_LLVM_HEADERS}
  COMMAND mkdir -p inc/llvm/include
  COMMAND mkdir -p inc/mlir/include
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Creating link to LLVM and MLIR headers"
  VERBATIM
)

if(ENABLE_INSTALLED_MLIR)
add_custom_target(HEIRIncludeLLVM
                  BYPRODUCTS ${CMAKE_BINARY_DIR}/inc/llvm/include/llvm
                  COMMAND [ -e llvm ] && echo "" || ln -s ${LLVM_INCLUDE_DIRS}/llvm/ ./
                  DEPENDS ${INCLUDE_MLIR_LLVM_HEADERS}
                  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/inc/llvm/include/"
                  )

add_custom_target(HEIRIncludeMLIR
                  BYPRODUCTS ${CMAKE_BINARY_DIR}/inc/mlir/include/mlir
                  COMMAND [ -e mlir ] && echo "" || ln -s ${MLIR_INCLUDE_DIRS}/mlir/ ./
                  DEPENDS ${INCLUDE_MLIR_LLVM_HEADERS}
                  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/inc/mlir/include/")
message(STATUS "Adding MLIR and LLVM includes.")
else()
add_custom_target(HEIRIncludeLLVM
          COMMAND echo "")
add_custom_target(HEIRIncludeMLIR
          COMMAND echo "")
message(STATUS "Using MLIR and LLVM from build folders of ${MLIR_DIR} and ${LLVM_DIR}")
endif()

include_directories(${CMAKE_BINARY_DIR}/inc)
if(ENABLE_INSTALLED_MLIR)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
else()
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS}/../..)
include_directories(${MLIR_INCLUDE_DIRS}/../..)
endif()
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_BINARY_DIR})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")
set(HEIRLIBDIR ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(tests)
