// Byte substitution function using Bayer-Poralta boolean method

func.func private @sub_byte(%x: i8 {secret.secret}) -> i8 {
    %c1 = arith.constant 1 : i8

    %c1_i8 = arith.constant 1 : i8
    %1 = arith.andi %x, %c1_i8 : i8
    %x7 = arith.trunci %1 : i8 to i1

    %2 = arith.shrsi %x, %c1 : i8
    %3 = arith.andi %2, %c1_i8 : i8
    %x6 = arith.trunci %3 : i8 to i1

    %4 = arith.shrsi %2, %c1 : i8
    %5 = arith.andi %4, %c1_i8 : i8
    %x5 = arith.trunci %5 : i8 to i1

    %6 = arith.shrsi %4, %c1 : i8
    %7 = arith.andi %6, %c1_i8 : i8
    %x4 = arith.trunci %7 : i8 to i1

    %8 = arith.shrsi %6, %c1 : i8
    %9 = arith.andi %8, %c1_i8 : i8
    %x3 = arith.trunci %9 : i8 to i1

    %10 = arith.shrsi %8, %c1 : i8
    %11 = arith.andi %10, %c1_i8 : i8
    %x2 = arith.trunci %11 : i8 to i1

    %12 = arith.shrsi %10, %c1 : i8
    %13 = arith.andi %12, %c1_i8 : i8
    %x1 = arith.trunci %13 : i8 to i1

    %14 = arith.shrsi %12, %c1 : i8
    %15 = arith.andi %14, %c1_i8 : i8
    %x0 = arith.trunci %15 : i8 to i1

    %45 = arith.xori %x3, %x5 : i1
    %46 = arith.xori %x0, %x6 : i1
    %47 = arith.xori %x0, %x3 : i1
    %48 = arith.xori %x0, %x5 : i1
    %49 = arith.xori %x1, %x2 : i1
    %51 = arith.xori %49, %x7 : i1
    %52 = arith.xori %51, %x3 : i1
    %53 = arith.xori %46, %45 : i1
    %54 = arith.xori %51, %x0 : i1
    %55 = arith.xori %51, %x6 : i1
    %56 = arith.xori %55, %48 : i1
    %57 = arith.xori %x4, %53 : i1
    %58 = arith.xori %57, %x5 : i1
    %59 = arith.xori %57, %x1 : i1
    %61 = arith.xori %58, %x7 : i1
    %62 = arith.xori %58, %49 : i1
    %63 = arith.xori %59, %47 : i1
    %65 = arith.xori %x7, %63 : i1
    %66 = arith.xori %62, %63 : i1
    %67 = arith.xori %62, %48 : i1
    %68 = arith.xori %49, %63 : i1
    %69 = arith.xori %46, %68 : i1
    %70 = arith.xori %x0, %68 : i1
    %71 = arith.andi %53, %58 : i1
    %72 = arith.andi %56, %61 : i1
    %73 = arith.xori %72, %71 : i1
    %75 = arith.andi %52, %x7 : i1
    %76 = arith.xori %75, %71 : i1
    %77 = arith.andi %46, %68 : i1
    %78 = arith.andi %55, %51 : i1
    %79 = arith.xori %78, %77 : i1
    %80 = arith.andi %54, %65 : i1
    %81 = arith.xori %80, %77 : i1
    %82 = arith.andi %47, %63 : i1
    %83 = arith.andi %45, %66 : i1
    %84 = arith.xori %83, %82 : i1
    %85 = arith.andi %48, %62 : i1
    %86 = arith.xori %85, %82 : i1
    %87 = arith.xori %73, %84 : i1
    %88 = arith.xori %76, %86 : i1
    %89 = arith.xori %79, %84 : i1
    %90 = arith.xori %81, %86 : i1
    %91 = arith.xori %87, %59 : i1
    %92 = arith.xori %88, %67 : i1
    %93 = arith.xori %89, %69 : i1
    %94 = arith.xori %90, %70 : i1
    %95 = arith.xori %91, %92 : i1
    %96 = arith.andi %91, %93 : i1
    %97 = arith.xori %94, %96 : i1
    %98 = arith.andi %95, %97 : i1
    %99 = arith.xori %98, %92 : i1
    %100 = arith.xori %93, %94 : i1
    %101 = arith.xori %92, %96 : i1
    %102 = arith.andi %101, %100 : i1
    %103 = arith.xori %102, %94 : i1
    %104 = arith.xori %93, %103 : i1
    %105 = arith.xori %97, %103 : i1
    %106 = arith.andi %94, %105 : i1
    %107 = arith.xori %106, %104 : i1
    %108 = arith.xori %97, %106 : i1
    %109 = arith.andi %99, %108 : i1
    %110 = arith.xori %95, %109 : i1
    %111 = arith.xori %110, %107 : i1
    %112 = arith.xori %99, %103 : i1
    %113 = arith.xori %99, %110 : i1
    %114 = arith.xori %103, %107 : i1
    %115 = arith.xori %112, %111 : i1
    %116 = arith.andi %114, %58 : i1
    %117 = arith.andi %107, %61 : i1
    %119 = arith.andi %103, %x7 : i1
    %120 = arith.andi %113, %68 : i1
    %121 = arith.andi %110, %51 : i1
    %122 = arith.andi %99, %65 : i1
    %123 = arith.andi %112, %63 : i1
    %124 = arith.andi %115, %66 : i1
    %125 = arith.andi %111, %62 : i1
    %126 = arith.andi %114, %53 : i1
    %127 = arith.andi %107, %56 : i1
    %128 = arith.andi %103, %52 : i1
    %129 = arith.andi %113, %46 : i1
    %130 = arith.andi %110, %55 : i1
    %131 = arith.andi %99, %54 : i1
    %132 = arith.andi %112, %47 : i1
    %133 = arith.andi %115, %45 : i1
    %134 = arith.andi %111, %48 : i1
    %135 = arith.xori %132, %133 : i1
    %136 = arith.xori %127, %128 : i1
    %137 = arith.xori %122, %130 : i1
    %138 = arith.xori %126, %127 : i1
    %139 = arith.xori %119, %129 : i1
    %140 = arith.xori %119, %122 : i1
    %141 = arith.xori %124, %125 : i1
    %142 = arith.xori %116, %120 : i1
    %143 = arith.xori %123, %124 : i1
    %144 = arith.xori %133, %134 : i1
    %145 = arith.xori %129, %137 : i1
    %146 = arith.xori %139, %142 : i1
    %147 = arith.xori %121, %135 : i1
    %148 = arith.xori %120, %143 : i1
    %149 = arith.xori %135, %146 : i1
    %150 = arith.xori %131, %146 : i1
    %151 = arith.xori %141, %147 : i1
    %152 = arith.xori %138, %147 : i1
    %153 = arith.xori %121, %148 : i1
    %154 = arith.xori %150, %151 : i1
    %155 = arith.xori %117, %152 : i1
    %156 = arith.xori %148, %152 : i1
    %157 = arith.constant 1  : i1
    %158 = arith.xori %157, %151 : i1
    %159 = arith.xori %145, %158 : i1
    %160 = arith.constant 1  : i1
    %161 = arith.xori %160, %149 : i1
    %162 = arith.xori %137, %161 : i1
    %163 = arith.xori %153, %154 : i1
    %164 = arith.xori %142, %155 : i1
    %165 = arith.xori %140, %155 : i1
    %166 = arith.xori %136, %154 : i1
    %167 = arith.constant 1  : i1
    %168 = arith.xori %167, %164 : i1
    %169 = arith.xori %153, %168 : i1
    %170 = arith.constant 1  : i1
    %171 = arith.xori %170, %163 : i1
    %172 = arith.xori %144, %171 : i1

    %s0 = arith.extui %156 : i1 to i8
    %c0_i8 = arith.constant 0 : i8
    %q = arith.addi %c0_i8, %s0 : i8

    %173 = arith.constant 1 : i8
    %175 = arith.shli %q, %173 : i8
    %s1 = arith.extui %169 : i1 to i8
    %176 = arith.addi %175, %s1 : i8
    %177 = arith.constant 1 : i8
    %179 = arith.shli %176, %177 : i8
    %s2 = arith.extui %172 : i1 to i8
    %180 = arith.addi %179, %s2 : i8
    %181 = arith.constant 1 : i8
    %183 = arith.shli %180, %181 : i8
    %s3 = arith.extui %164 : i1 to i8
    %184 = arith.addi %183, %s3 : i8
    %185 = arith.constant 1 : i8
    %187 = arith.shli %184, %185 : i8
    %s4 = arith.extui %165 : i1 to i8
    %188 = arith.addi %187, %s4 : i8
    %189 = arith.constant 1 : i8
    %191 = arith.shli %188, %189 : i8
    %s5 = arith.extui %166 : i1 to i8
    %192 = arith.addi %191, %s5 : i8
    %193 = arith.constant 1 : i8
    %195 = arith.shli %192, %193 : i8
    %s6 = arith.extui %159 : i1 to i8
    %196 = arith.addi %195, %s6 : i8
    %197 = arith.constant 1 : i8
    %199 = arith.shli %196, %197 : i8
    %s7 = arith.extui %162 : i1 to i8
    %200 = arith.addi %199, %s7 : i8
    func.return %200 : i8
}

#map = affine_map<(d0) -> (d0)>
func.func @sub_bytes(%arg0: tensor<16xi8> {secret.secret}) -> tensor<16xi8> {
  %0 = tensor.empty() : tensor<16xi8>
  %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : tensor<16xi8>) outs(%0 : tensor<16xi8>) {
  ^bb0(%in: i8, %out: i8):
    %extracted = func.call @sub_byte(%in) : (i8) -> i8
    linalg.yield %extracted : i8
  } -> tensor<16xi8>
  return %1 : tensor<16xi8>
}

// The lookup table implementation is commented below

// #map = affine_map<(d0) -> (d0)>
// module {
//   func.func @sub_bytes(%arg0: tensor<16xi8> {secret.secret}) -> tensor<16xi8> {
//     %cst = arith.constant dense<"0x637C777BF26B6FC53001672BFED7AB76CA82C97DFA5947F0ADD4A2AF9CA472C0B7FD9326363FF7CC34A5E5F171D8311504C723C31896059A071280E2EB27B27509832C1A1B6E5AA0523BD6B329E32F8453D100ED20FCB15B6ACBBE394A4C58CFD0EFAAFB434D338545F9027F503C9FA851A3408F929D38F5BCB6DA2110FFF3D2CD0C13EC5F974417C4A77E3D645D197360814FDC222A908846EEB814DE5E0BDBE0323A0A4906245CC2D3AC629195E479E7C8376D8DD54EA96C56F4EA657AAE08BA78252E1CA6B4C6E8DD741F4BBD8B8A703EB5664803F60E613557B986C11D9EE1F8981169D98E949B1E87E9CE5528DF8CA1890DBFE6426841992D0FB054BB16"> : tensor<256xi8>
//     %0 = tensor.empty() : tensor<16xi8>
//     %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : tensor<16xi8>) outs(%0 : tensor<16xi8>) {
//     ^bb0(%in: i8, %out: i8):
//       %2 = arith.index_cast %in : i8 to index
//       %extracted = tensor.extract %cst[%2] : tensor<256xi8>
//       linalg.yield %extracted : i8
//     } -> tensor<16xi8>
//     return %1 : tensor<16xi8>
//   }
// }
