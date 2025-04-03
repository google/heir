// RUN: heir-opt --forward-store-to-load %s | FileCheck %s

module {
  // CHECK: @add_one
  func.func @add_one(%arg0: !tfhe_rust.server_key, %arg1: memref<8x!tfhe_rust.eui3>) -> memref<8x!tfhe_rust.eui3> {
    %c1_i8 = arith.constant 1 : i8
    %c2_i8 = arith.constant 2 : i8
    %true = arith.constant true
    %false = arith.constant false
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    // This alloc is not needed, so we should test that the stores are all forwarded to their loads.
    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<8xi1>
    // CHECK-NOT: memref.load %[[ALLOC]]
    %alloc = memref.alloc() : memref<8xi1>
    memref.store %true, %alloc[%c0] : memref<8xi1>
    memref.store %false, %alloc[%c1] : memref<8xi1>
    memref.store %false, %alloc[%c2] : memref<8xi1>
    memref.store %false, %alloc[%c3] : memref<8xi1>
    memref.store %false, %alloc[%c4] : memref<8xi1>
    memref.store %false, %alloc[%c5] : memref<8xi1>
    memref.store %false, %alloc[%c6] : memref<8xi1>
    memref.store %false, %alloc[%c7] : memref<8xi1>
    %0 = memref.load %alloc[%c0] : memref<8xi1>
    %1 = memref.load %arg1[%c0] : memref<8x!tfhe_rust.eui3>
    %2 = tfhe_rust.create_trivial %arg0, %false : (!tfhe_rust.server_key, i1) -> !tfhe_rust.eui3
    %3 = tfhe_rust.create_trivial %arg0, %0 : (!tfhe_rust.server_key, i1) -> !tfhe_rust.eui3
    %4 = tfhe_rust.generate_lookup_table %arg0 {truthTable = 8 : ui8} : (!tfhe_rust.server_key) -> !tfhe_rust.lookup_table
    %5 = tfhe_rust.scalar_left_shift %arg0, %2 {shiftAmount = 2 : index} : (!tfhe_rust.server_key, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %6 = tfhe_rust.scalar_left_shift %arg0, %3 {shiftAmount = 1 : index} : (!tfhe_rust.server_key, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %7 = tfhe_rust.add %arg0, %5, %6 : (!tfhe_rust.server_key, !tfhe_rust.eui3, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %8 = tfhe_rust.add %arg0, %7, %1 : (!tfhe_rust.server_key, !tfhe_rust.eui3, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %9 = tfhe_rust.apply_lookup_table %arg0, %8, %4 : (!tfhe_rust.server_key, !tfhe_rust.eui3, !tfhe_rust.lookup_table) -> !tfhe_rust.eui3
    %10 = memref.load %alloc[%c1] : memref<8xi1>
    %11 = memref.load %arg1[%c1] : memref<8x!tfhe_rust.eui3>
    %12 = tfhe_rust.create_trivial %arg0, %10 : (!tfhe_rust.server_key, i1) -> !tfhe_rust.eui3
    %13 = tfhe_rust.generate_lookup_table %arg0 {truthTable = 150 : ui8} : (!tfhe_rust.server_key) -> !tfhe_rust.lookup_table
    %14 = tfhe_rust.scalar_left_shift %arg0, %12 {shiftAmount = 2 : index} : (!tfhe_rust.server_key, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %15 = tfhe_rust.scalar_left_shift %arg0, %11 {shiftAmount = 1 : index} : (!tfhe_rust.server_key, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %16 = tfhe_rust.add %arg0, %14, %15 : (!tfhe_rust.server_key, !tfhe_rust.eui3, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %17 = tfhe_rust.add %arg0, %16, %9 : (!tfhe_rust.server_key, !tfhe_rust.eui3, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %18 = tfhe_rust.apply_lookup_table %arg0, %17, %13 : (!tfhe_rust.server_key, !tfhe_rust.eui3, !tfhe_rust.lookup_table) -> !tfhe_rust.eui3
    %19 = tfhe_rust.generate_lookup_table %arg0 {truthTable = 23 : ui8} : (!tfhe_rust.server_key) -> !tfhe_rust.lookup_table
    %20 = tfhe_rust.apply_lookup_table %arg0, %17, %19 : (!tfhe_rust.server_key, !tfhe_rust.eui3, !tfhe_rust.lookup_table) -> !tfhe_rust.eui3
    %21 = memref.load %alloc[%c2] : memref<8xi1>
    %22 = memref.load %arg1[%c2] : memref<8x!tfhe_rust.eui3>
    %23 = tfhe_rust.create_trivial %arg0, %21 : (!tfhe_rust.server_key, i1) -> !tfhe_rust.eui3
    %24 = tfhe_rust.generate_lookup_table %arg0 {truthTable = 43 : ui8} : (!tfhe_rust.server_key) -> !tfhe_rust.lookup_table
    %25 = tfhe_rust.scalar_left_shift %arg0, %23 {shiftAmount = 2 : index} : (!tfhe_rust.server_key, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %26 = tfhe_rust.scalar_left_shift %arg0, %22 {shiftAmount = 1 : index} : (!tfhe_rust.server_key, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %27 = tfhe_rust.add %arg0, %25, %26 : (!tfhe_rust.server_key, !tfhe_rust.eui3, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %28 = tfhe_rust.add %arg0, %27, %20 : (!tfhe_rust.server_key, !tfhe_rust.eui3, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %29 = tfhe_rust.apply_lookup_table %arg0, %28, %24 : (!tfhe_rust.server_key, !tfhe_rust.eui3, !tfhe_rust.lookup_table) -> !tfhe_rust.eui3
    %30 = memref.load %alloc[%c3] : memref<8xi1>
    %31 = memref.load %arg1[%c3] : memref<8x!tfhe_rust.eui3>
    %32 = tfhe_rust.create_trivial %arg0, %30 : (!tfhe_rust.server_key, i1) -> !tfhe_rust.eui3
    %33 = tfhe_rust.scalar_left_shift %arg0, %32 {shiftAmount = 2 : index} : (!tfhe_rust.server_key, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %34 = tfhe_rust.scalar_left_shift %arg0, %31 {shiftAmount = 1 : index} : (!tfhe_rust.server_key, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %35 = tfhe_rust.add %arg0, %33, %34 : (!tfhe_rust.server_key, !tfhe_rust.eui3, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %36 = tfhe_rust.add %arg0, %35, %29 : (!tfhe_rust.server_key, !tfhe_rust.eui3, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %37 = tfhe_rust.apply_lookup_table %arg0, %36, %24 : (!tfhe_rust.server_key, !tfhe_rust.eui3, !tfhe_rust.lookup_table) -> !tfhe_rust.eui3
    %38 = memref.load %alloc[%c4] : memref<8xi1>
    %39 = memref.load %arg1[%c4] : memref<8x!tfhe_rust.eui3>
    %40 = tfhe_rust.create_trivial %arg0, %38 : (!tfhe_rust.server_key, i1) -> !tfhe_rust.eui3
    %41 = tfhe_rust.scalar_left_shift %arg0, %40 {shiftAmount = 2 : index} : (!tfhe_rust.server_key, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %42 = tfhe_rust.scalar_left_shift %arg0, %39 {shiftAmount = 1 : index} : (!tfhe_rust.server_key, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %43 = tfhe_rust.add %arg0, %41, %42 : (!tfhe_rust.server_key, !tfhe_rust.eui3, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %44 = tfhe_rust.add %arg0, %43, %37 : (!tfhe_rust.server_key, !tfhe_rust.eui3, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %45 = tfhe_rust.apply_lookup_table %arg0, %44, %24 : (!tfhe_rust.server_key, !tfhe_rust.eui3, !tfhe_rust.lookup_table) -> !tfhe_rust.eui3
    %46 = memref.load %alloc[%c5] : memref<8xi1>
    %47 = memref.load %arg1[%c5] : memref<8x!tfhe_rust.eui3>
    %48 = tfhe_rust.create_trivial %arg0, %46 : (!tfhe_rust.server_key, i1) -> !tfhe_rust.eui3
    %49 = tfhe_rust.scalar_left_shift %arg0, %48 {shiftAmount = 2 : index} : (!tfhe_rust.server_key, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %50 = tfhe_rust.scalar_left_shift %arg0, %47 {shiftAmount = 1 : index} : (!tfhe_rust.server_key, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %51 = tfhe_rust.add %arg0, %49, %50 : (!tfhe_rust.server_key, !tfhe_rust.eui3, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %52 = tfhe_rust.add %arg0, %51, %45 : (!tfhe_rust.server_key, !tfhe_rust.eui3, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %53 = tfhe_rust.apply_lookup_table %arg0, %52, %24 : (!tfhe_rust.server_key, !tfhe_rust.eui3, !tfhe_rust.lookup_table) -> !tfhe_rust.eui3
    %54 = memref.load %alloc[%c6] : memref<8xi1>
    %55 = memref.load %arg1[%c6] : memref<8x!tfhe_rust.eui3>
    %56 = tfhe_rust.create_trivial %arg0, %54 : (!tfhe_rust.server_key, i1) -> !tfhe_rust.eui3
    %57 = tfhe_rust.generate_lookup_table %arg0 {truthTable = 105 : ui8} : (!tfhe_rust.server_key) -> !tfhe_rust.lookup_table
    %58 = tfhe_rust.scalar_left_shift %arg0, %56 {shiftAmount = 2 : index} : (!tfhe_rust.server_key, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %59 = tfhe_rust.scalar_left_shift %arg0, %55 {shiftAmount = 1 : index} : (!tfhe_rust.server_key, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %60 = tfhe_rust.add %arg0, %58, %59 : (!tfhe_rust.server_key, !tfhe_rust.eui3, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %61 = tfhe_rust.add %arg0, %60, %53 : (!tfhe_rust.server_key, !tfhe_rust.eui3, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %62 = tfhe_rust.apply_lookup_table %arg0, %61, %57 : (!tfhe_rust.server_key, !tfhe_rust.eui3, !tfhe_rust.lookup_table) -> !tfhe_rust.eui3
    %63 = tfhe_rust.apply_lookup_table %arg0, %61, %24 : (!tfhe_rust.server_key, !tfhe_rust.eui3, !tfhe_rust.lookup_table) -> !tfhe_rust.eui3
    %64 = memref.load %alloc[%c7] : memref<8xi1>
    %65 = memref.load %arg1[%c7] : memref<8x!tfhe_rust.eui3>
    %66 = tfhe_rust.create_trivial %arg0, %64 : (!tfhe_rust.server_key, i1) -> !tfhe_rust.eui3
    %67 = tfhe_rust.scalar_left_shift %arg0, %66 {shiftAmount = 2 : index} : (!tfhe_rust.server_key, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %68 = tfhe_rust.scalar_left_shift %arg0, %65 {shiftAmount = 1 : index} : (!tfhe_rust.server_key, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %69 = tfhe_rust.add %arg0, %67, %68 : (!tfhe_rust.server_key, !tfhe_rust.eui3, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %70 = tfhe_rust.add %arg0, %69, %63 : (!tfhe_rust.server_key, !tfhe_rust.eui3, !tfhe_rust.eui3) -> !tfhe_rust.eui3
    %71 = tfhe_rust.apply_lookup_table %arg0, %70, %57 : (!tfhe_rust.server_key, !tfhe_rust.eui3, !tfhe_rust.lookup_table) -> !tfhe_rust.eui3
    %72 = tfhe_rust.generate_lookup_table %arg0 {truthTable = 6 : ui8} : (!tfhe_rust.server_key) -> !tfhe_rust.lookup_table
    %73 = tfhe_rust.apply_lookup_table %arg0, %8, %72 : (!tfhe_rust.server_key, !tfhe_rust.eui3, !tfhe_rust.lookup_table) -> !tfhe_rust.eui3
    %74 = tfhe_rust.apply_lookup_table %arg0, %28, %57 : (!tfhe_rust.server_key, !tfhe_rust.eui3, !tfhe_rust.lookup_table) -> !tfhe_rust.eui3
    %75 = tfhe_rust.apply_lookup_table %arg0, %36, %57 : (!tfhe_rust.server_key, !tfhe_rust.eui3, !tfhe_rust.lookup_table) -> !tfhe_rust.eui3
    %76 = tfhe_rust.apply_lookup_table %arg0, %44, %57 : (!tfhe_rust.server_key, !tfhe_rust.eui3, !tfhe_rust.lookup_table) -> !tfhe_rust.eui3
    %77 = tfhe_rust.apply_lookup_table %arg0, %52, %57 : (!tfhe_rust.server_key, !tfhe_rust.eui3, !tfhe_rust.lookup_table) -> !tfhe_rust.eui3
    %alloc_0 = memref.alloc() : memref<8x!tfhe_rust.eui3>
    memref.store %73, %alloc_0[%c0] : memref<8x!tfhe_rust.eui3>
    memref.store %18, %alloc_0[%c1] : memref<8x!tfhe_rust.eui3>
    memref.store %74, %alloc_0[%c2] : memref<8x!tfhe_rust.eui3>
    memref.store %75, %alloc_0[%c3] : memref<8x!tfhe_rust.eui3>
    memref.store %76, %alloc_0[%c4] : memref<8x!tfhe_rust.eui3>
    memref.store %77, %alloc_0[%c5] : memref<8x!tfhe_rust.eui3>
    memref.store %62, %alloc_0[%c6] : memref<8x!tfhe_rust.eui3>
    memref.store %71, %alloc_0[%c7] : memref<8x!tfhe_rust.eui3>
    return %alloc_0 : memref<8x!tfhe_rust.eui3>
  }
}
