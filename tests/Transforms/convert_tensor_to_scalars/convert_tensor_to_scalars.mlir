// RUN: heir-opt --convert-tensor-to-scalars %s | FileCheck --check-prefix=COMMONCHECK --check-prefix=CHECK %s
// RUN: heir-opt --convert-tensor-to-scalars=max-size=0 %s | FileCheck --check-prefix=COMMONCHECK --check-prefix=NOOPCHECK %s
!t = tensor<2xi32>

//COMMONCHECK: @test_fn
//CHECK: [[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32
//NOOPCHECK: [[ARG0:%.+]]: tensor<2xi32>
func.func @test_fn(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    //CHECK: return [[ARG0]], [[ARG1]] : i32, i32
    //NOOPCHECK: return [[ARG0]] : tensor<2xi32>
    return %arg0 : tensor<2xi32>
}

//COMMONCHECK: @test_scf
//CHECK: [[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32, [[ARG2:%.+]]: i32, [[ARG3:%.+]]: i32, [[COND:%.+]]: i1
//NOOPCHECK: [[ARG0:%.+]]: tensor<2xi32>, [[ARG1:%.+]]: tensor<2xi32>, [[COND:%.+]]: i1
func.func @test_scf(%arg0: tensor<2xi32>, %arg1 : tensor<2xi32>, %cond : i1) -> tensor<2xi32> {
    //CHECK: [[IF:%.+]]:2 = scf.if [[COND]] -> (i32, i32)
    //NOOPCHECK: [[IF:%.+]] = scf.if [[COND]] -> (tensor<2xi32>)
    %0 = scf.if %cond -> (tensor<2xi32>) {
        //CHECK: scf.yield [[ARG0]], [[ARG1]] : i32, i32
        //NOOPCHECK: scf.yield [[ARG0]] : tensor<2xi32>
        scf.yield %arg0 : tensor<2xi32>
    } else {
        //CHECK: scf.yield [[ARG2]], [[ARG3]] : i32, i32
        //NOOPCHECK: scf.yield [[ARG1]] : tensor<2xi32>
        scf.yield %arg1 : tensor<2xi32>
    }
    //CHECK: return [[IF]]#0, [[IF]]#1 : i32, i32
    //NOOPCHECK: return [[IF]] : tensor<2xi32>
    return %0 : tensor<2xi32>
}

//COMMONCHECK: @test_dyn_noop
//COMMONCHECK: [[ARG0:%.+]]: tensor<?xi32>
func.func @test_dyn_noop(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    //COMMONCHECK: return [[ARG0]] : tensor<?xi32>
    return %arg0 : tensor<?xi32>
}

//COMMONCHECK: @test_extract_insert
//CHECK: [[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32, [[ARG2:%.+]]: i32, [[ARG3:%.+]]: i32
//NOOPCHECK: [[ARG0:%.+]]: tensor<2xi32>, [[ARG1:%.+]]: tensor<2xi32>
func.func @test_extract_insert(%arg0: tensor<2xi32>, %arg1 : tensor<2xi32>) -> tensor<2xi32> {
    //CHECK-NOT: arith.constant
    //NOOPCHECK: [[C0:%.+]] = arith.constant 0 : index
    %c0 = arith.constant 0 : index
    //NOOPCHECK: [[C1:%.+]] = arith.constant 1 : index
    %c1 = arith.constant 1 : index
    //CHECK-NOT: tensor.empty
    //NOOPCHECK: [[T:%.+]] = tensor.empty() : tensor<2xi32>
    %t = tensor.empty() : tensor<2xi32>
    //CHECK-NOT: tensor.extract
    //NOOPCHECK: [[X0:%.+]] = tensor.extract [[ARG0]][[[C0]]] : tensor<2xi32>
    %x0 = tensor.extract %arg0[%c0] : tensor<2xi32>
    //NOOPCHECK: [[Y0:%.+]] = tensor.extract [[ARG1]][[[C0]]] : tensor<2xi32>
    %y0 = tensor.extract %arg1[%c0] : tensor<2xi32>
    //CHECK: [[ADD0:%.+]] = arith.addi [[ARG0]], [[ARG2]] : i32
    //NOOPCHECK: [[ADD0:%.+]] = arith.addi [[X0]], [[Y0]] : i32
    %a0 = arith.addi %x0, %y0: i32
    //CHECK-NOT: tensor.insert
    //NOOPCHECK: [[TT:%.+]] = tensor.insert [[ADD0]] into [[T]][[[C0]]] : tensor<2xi32>
    %tt = tensor.insert %a0 into %t[%c0] : tensor<2xi32>
    //CHECK-NOT: tensor.extract
    //NOOPCHECK: [[X1:%.+]] = tensor.extract [[ARG0]][[[C1]]] : tensor<2xi32>
    %x1 = tensor.extract %arg0[%c1] : tensor<2xi32>
    //NOOPCHECK: [[Y1:%.+]] = tensor.extract [[ARG1]][[[C1]]] : tensor<2xi32>
    %y1 = tensor.extract %arg1[%c1] : tensor<2xi32>
    //CHECK: [[ADD1:%.+]] = arith.addi [[ARG1]], [[ARG3]] : i32
    //NOOPCHECK: [[ADD1:%.+]] = arith.addi [[X1]], [[Y1]] : i32
    %a1 = arith.addi %x1, %y1: i32
    //CHECK-NOT: tensor.insert
    //NOOPCHECK: [[TTT:%.+]] = tensor.insert [[ADD1]] into [[TT]][[[C1]]] : tensor<2xi32>
    %ttt = tensor.insert %a1 into %tt[%c1] : tensor<2xi32>
    //CHECK: return [[ADD0]], [[ADD1]] : i32, i32
    //NOOPCHECK: return [[TTT]] : tensor<2xi32>
    return %ttt : tensor<2xi32>
}

//COMMONCHECK: @test_materialize
//CHECK: [[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32, [[ARG2:%.+]]: i32, [[ARG3:%.+]]: i32
//NOOPCHECK: [[ARG0:%.+]]: tensor<2xi32>, [[ARG1:%.+]]: tensor<2xi32>
func.func @test_materialize(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi32> {
    //CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
    //CHECK-DAG: [[C1:%.+]] = arith.constant 1 : index
    //CHECK-DAG: [[T0:%.+]] = tensor.from_elements [[ARG0]], [[ARG1]] : tensor<2xi32>
    //CHECK-DAG: [[T1:%.+]] = tensor.from_elements [[ARG2]], [[ARG3]] : tensor<2xi32>
    //CHECK: [[ADD:%.+]] = arith.addi [[T0]], [[T1]] : tensor<2xi32>
    //NOOPCHECK: [[ADD:%.+]] = arith.addi [[ARG0]], [[ARG1]] : tensor<2xi32>
    %0 = arith.addi %arg0, %arg1 : tensor<2xi32>
    //CHECK-DAG: [[ADD0:%.+]] = tensor.extract [[ADD]][[[C0]]] : tensor<2xi32>
    //CHECK-DAG: [[ADD1:%.+]] = tensor.extract [[ADD]][[[C1]]] : tensor<2xi32>
    //CHECK: return [[ADD0]], [[ADD1]] : i32, i32
    //NOOPCHECK: return [[ADD]] : tensor<2xi32>
    return %0 : tensor<2xi32>
}
