//===- CombDialect.cpp - Implement the Comb dialect -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Comb dialect.
//
//===----------------------------------------------------------------------===//

#include "include/Dialect/Comb/IR/CombDialect.h"

#include "include/Dialect/Comb/IR/CombOps.h"
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace comb {

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void CombDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "include/Dialect/Comb/IR/Comb.cpp.inc"
      >();
}

}  // namespace comb
}  // namespace heir
}  // namespace mlir

// Provide implementations for the enums we use.
#include "include/Dialect/Comb/IR/CombDialect.cpp.inc"
#include "include/Dialect/Comb/IR/CombEnums.cpp.inc"
