// RUN: heir-translate --emit-pisa %s | FileCheck %s

!m32 = !mod_arith.int<33538049:i32>

func.func @test_emit(%arg0 : tensor<8192x!m32>, %arg1 : tensor<8192x!m32>) ->  tensor<8192x!m32> {
    //CHECK: 13, add, [[ADD:.+]], [[INP0:.+]], [[INP1:.+]], 0
    //CHECK: 13, sub, [[SUB:.+]], [[INP0]], [[INP1]], 0
    //CHECK: 13, mul, [[MUL:.+]], [[INP0]], [[INP1]], 0
    //CHECK: 13, mul, [[MULI:.+]], [[INP0]], [[MULI]]_imm, 0
    //CHECK: 13, copy, [[ACC1:.+]], [[INP0]]
    //CHECK: 13, mac, [[ACC1]], [[INP0]], [[INP1]], 0
    //CHECK: 13, copy, [[ACC2:.+]], [[INP1]]
    //CHECK: 13, mac, [[ACC2]], [[INP0]], [[ACC2]]_imm, 0
    %0 = pisa.add %arg0, %arg1 {q = 2147483647 : i32, i = 0 : i32} : tensor<8192x!m32>
    %1 = pisa.sub %arg0, %arg1 {q = 2147483647 : i32, i = 0 : i32} : tensor<8192x!m32>
    %2 = pisa.mul %arg0, %arg1 {q = 2147483647 : i32, i = 0 : i32} : tensor<8192x!m32>
    %3 = pisa.muli %arg0 {q = 2147483647 : i32, i = 0 : i32, imm = 5 : i32} : tensor<8192x!m32>
    %4 = pisa.mac %arg0, %arg1, %arg0 {q = 2147483647 : i32, i = 0 : i32} : tensor<8192x!m32>
    %5 = pisa.maci %arg0, %arg1 {q = 2147483647 : i32, i = 0 : i32, imm = 5 : i32} : tensor<8192x!m32>
    // %w = mod_arith.constant 42 : tensor<8192x!m32> // FIXME: re-enable once mod_arith tensor constant generation is fixed
    // %6 = pisa.ntt %arg0, %w {q = 2147483647 : i32, i = 0 : i32} : tensor<8192x!m32>
    // %7 = pisa.intt %arg0, %w {q = 2147483647 : i32, i = 0 : i32} : tensor<8192x!m32>
    return %0 : tensor<8192x!m32>
}
