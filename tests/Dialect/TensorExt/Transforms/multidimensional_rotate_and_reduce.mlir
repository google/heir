// RUN: heir-opt --rotate-and-reduce --cse --canonicalize %s | FileCheck %s

// Regression for #1039: multidimensional tensors are not supported in RotationAnalysis.

#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  // CHECK: @matmul
  // CHECK-COUNT-15: tensor_ext.rotate
  // CHECK: return
  func.func @matmul(%arg0: !secret.secret<tensor<1x16xf32>>) -> !secret.secret<tensor<1x16xf32>> {
    %cst = arith.constant dense<[-0.45141533, -0.0277900472, 0.311195374, 0.18254894, -0.258809537, 0.497506738, 0.00115649134, -0.194445714, 0.158549473, 0.000000e+00, 0.310650676, -0.214976981, -0.023661999, -0.392960966, 6.472870e-01, 0.831665277]> : tensor<16xf32>
    %cst_0 = arith.constant dense<"0x5036CB3D584F72BFBA00FA3DBF82CB3FBF6FFB3F6D8B66BDAE8AA13EBF5F023FC2D49C3FCCA8D6BD532AA23F522F40BE1AD4EABE79B7B8BE37B836BF1C591EBEED693F3E27116D3D0DBFF0BE726237BF9C8203BF4DF072BFAF6CACBD83868D3F8523E03DD65AFDBE15A656BE8B94383DC5DE023F4A69D73E989532BE66736C3E647E8E3FAA05383F79DF193FE1645E3F24B92ABEC166A93E615C2DC0497814BD07ED413DD757033F1923163D76727CBEC759C83FF01E2F3D23B418BFABF141BE1C7029BF32109FBF4E00073FA4569FBD7F70C5BE8D4BDF3F473F593F911CAB3F13E9853E590DF63EC24284BEADBB19BF37CE383FEB0713BDD8C9243F63FE853EBEE1923FC956703F29DA10BEB0863BBF733F6C3FBF4AEABDE32D25BF4BC424BE016DA2BEC4280CBF374FFD3D755B6F3F9B9C1BBECBACC6BD6CAA17BE26F7803F83F258BEADE2DCBE1C9693BDD15A1FC07CE7DA3E9FD976BFDF71D0BD7A020C3FD73A6F3E8EA7EABEB9F2A3BED185903E4C08633DCA2E0EBF21AE853F68822ABF77DDA1BFEE8F443E12360BBF97BE063E1F83C13FB809763E382D6CBE6509A73E2D0268BE74C317BE4597B5BB6294083F195E3E3E28F39CBC3C11793F7BFA0F3FD7EDB93C3EF9903FEEFCFBBD94A451BF0284113ECD4C10BF2DD392BD95E499BEEBC9613EBEB947BE576920BF8D0700BF90944FBDE8A4C23FBE51FABE34128B3E309239C090A07CBF051EBDBEE4AA0B3FF339193FC88265BF9C4FECB8EE54DB3EEFFC3CBF870B93BE6A4E0F3ED998223F511F9DBF8D844E3F47B452BEA3B655BF842D833E7803133FAE500E3F0C8726BE1B19E13C6B2BABBE4BF853BED6E0493E695FA93E3402CB3EEC66B9BEACE7A1BF8E44723F2791AF3D2A0A763EAB2F1C3EB0C3163FCDB2E83E598190BFA2AD41BEA1F5FA3EC202C5BE8D53EB3DCDCA91BEA695923EC68B41BE4072EE3F79CC633C665C36BFF31C79BE4CA4BDBFCC272F3FD1DB2ABFDC29103F369C0D3F1A15F63E6E31C9BEA1078D3EBE94AABE387D02BF1937CB3F50F054BF570AFE3D1039C43DCE54193F5129963F7AE739BE2B3A19BF5D971B3FCC190A3F65AB74BF1CDB073F40F376BF5A0439BF3C3297BDF615053D553A37BF161EB23E961D7C3E8FB21CBF9ABE19C074A657BF6C0AC33F198BD9BE1091853EECD986BF022E55BE47C021BE4D87BDBE1FD4A7BE75BF7FBE7D4EB63F6EC745BF4FDC1F3E5FEE0ABD36CD1B3FECACB53E3D09BD3D0FDF2E3DA75C0DBFED900CBEAE4E853DE4A9393EBD0E8D3EA96AA03E3640A63F1EFEF83E0042113E3DCF823EB562A1BD1EC09EBF99D43DBE88925B3EF1F9953EBDA0BABC12C739BD19464BBFE147C3BE3CB9E83EDDD4C93E8D9F843D55B69FBF1936893F4C661B3FEC98153FD9A115C0EA77253E85E451BFA19AA4BEFDA50DBB6E8E23BFE5F4E1BE"> : tensor<16x16xf32>
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %c9 = arith.constant 9 : index
    %c10 = arith.constant 10 : index
    %c11 = arith.constant 11 : index
    %c12 = arith.constant 12 : index
    %c13 = arith.constant 13 : index
    %c14 = arith.constant 14 : index
    %c15 = arith.constant 15 : index
    %0 = secret.generic {
      %2 = tensor.empty() : tensor<1x16xf32>
      secret.yield %2 : tensor<1x16xf32>
    } -> !secret.secret<tensor<1x16xf32>>
    %extracted_slice = tensor.extract_slice %cst_0[0, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
    %extracted_slice_1 = tensor.extract_slice %cst_0[1, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
    %extracted_slice_2 = tensor.extract_slice %cst_0[2, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
    %extracted_slice_3 = tensor.extract_slice %cst_0[3, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
    %extracted_slice_4 = tensor.extract_slice %cst_0[4, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
    %extracted_slice_5 = tensor.extract_slice %cst_0[5, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
    %extracted_slice_6 = tensor.extract_slice %cst_0[6, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
    %extracted_slice_7 = tensor.extract_slice %cst_0[7, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
    %extracted_slice_8 = tensor.extract_slice %cst_0[8, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
    %extracted_slice_9 = tensor.extract_slice %cst_0[9, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
    %extracted_slice_10 = tensor.extract_slice %cst_0[10, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
    %extracted_slice_11 = tensor.extract_slice %cst_0[11, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
    %extracted_slice_12 = tensor.extract_slice %cst_0[12, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
    %extracted_slice_13 = tensor.extract_slice %cst_0[13, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
    %extracted_slice_14 = tensor.extract_slice %cst_0[14, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
    %extracted_slice_15 = tensor.extract_slice %cst_0[15, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
    %1 = secret.generic ins(%arg0, %0 : !secret.secret<tensor<1x16xf32>>, !secret.secret<tensor<1x16xf32>>) {
    ^bb0(%arg1: tensor<1x16xf32>, %arg2: tensor<1x16xf32>):
      %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst : tensor<16xf32>) outs(%arg2 : tensor<1x16xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<1x16xf32>
      %3 = arith.mulf %arg1, %extracted_slice : tensor<1x16xf32>
      %4 = arith.addf %2, %3 : tensor<1x16xf32>
      %5 = tensor_ext.rotate %arg1, %c1 : tensor<1x16xf32>, index
      %6 = arith.mulf %5, %extracted_slice_1 : tensor<1x16xf32>
      %7 = arith.addf %4, %6 : tensor<1x16xf32>
      %8 = tensor_ext.rotate %arg1, %c2 : tensor<1x16xf32>, index
      %9 = arith.mulf %8, %extracted_slice_2 : tensor<1x16xf32>
      %10 = arith.addf %7, %9 : tensor<1x16xf32>
      %11 = tensor_ext.rotate %arg1, %c3 : tensor<1x16xf32>, index
      %12 = arith.mulf %11, %extracted_slice_3 : tensor<1x16xf32>
      %13 = arith.addf %10, %12 : tensor<1x16xf32>
      %14 = tensor_ext.rotate %arg1, %c4 : tensor<1x16xf32>, index
      %15 = arith.mulf %14, %extracted_slice_4 : tensor<1x16xf32>
      %16 = arith.addf %13, %15 : tensor<1x16xf32>
      %17 = tensor_ext.rotate %arg1, %c5 : tensor<1x16xf32>, index
      %18 = arith.mulf %17, %extracted_slice_5 : tensor<1x16xf32>
      %19 = arith.addf %16, %18 : tensor<1x16xf32>
      %20 = tensor_ext.rotate %arg1, %c6 : tensor<1x16xf32>, index
      %21 = arith.mulf %20, %extracted_slice_6 : tensor<1x16xf32>
      %22 = arith.addf %19, %21 : tensor<1x16xf32>
      %23 = tensor_ext.rotate %arg1, %c7 : tensor<1x16xf32>, index
      %24 = arith.mulf %23, %extracted_slice_7 : tensor<1x16xf32>
      %25 = arith.addf %22, %24 : tensor<1x16xf32>
      %26 = tensor_ext.rotate %arg1, %c8 : tensor<1x16xf32>, index
      %27 = arith.mulf %26, %extracted_slice_8 : tensor<1x16xf32>
      %28 = arith.addf %25, %27 : tensor<1x16xf32>
      %29 = tensor_ext.rotate %arg1, %c9 : tensor<1x16xf32>, index
      %30 = arith.mulf %29, %extracted_slice_9 : tensor<1x16xf32>
      %31 = arith.addf %28, %30 : tensor<1x16xf32>
      %32 = tensor_ext.rotate %arg1, %c10 : tensor<1x16xf32>, index
      %33 = arith.mulf %32, %extracted_slice_10 : tensor<1x16xf32>
      %34 = arith.addf %31, %33 : tensor<1x16xf32>
      %35 = tensor_ext.rotate %arg1, %c11 : tensor<1x16xf32>, index
      %36 = arith.mulf %35, %extracted_slice_11 : tensor<1x16xf32>
      %37 = arith.addf %34, %36 : tensor<1x16xf32>
      %38 = tensor_ext.rotate %arg1, %c12 : tensor<1x16xf32>, index
      %39 = arith.mulf %38, %extracted_slice_12 : tensor<1x16xf32>
      %40 = arith.addf %37, %39 : tensor<1x16xf32>
      %41 = tensor_ext.rotate %arg1, %c13 : tensor<1x16xf32>, index
      %42 = arith.mulf %41, %extracted_slice_13 : tensor<1x16xf32>
      %43 = arith.addf %40, %42 : tensor<1x16xf32>
      %44 = tensor_ext.rotate %arg1, %c14 : tensor<1x16xf32>, index
      %45 = arith.mulf %44, %extracted_slice_14 : tensor<1x16xf32>
      %46 = arith.addf %43, %45 : tensor<1x16xf32>
      %47 = tensor_ext.rotate %arg1, %c15 : tensor<1x16xf32>, index
      %48 = arith.mulf %47, %extracted_slice_15 : tensor<1x16xf32>
      %49 = arith.addf %46, %48 : tensor<1x16xf32>
      secret.yield %49 : tensor<1x16xf32>
    } -> !secret.secret<tensor<1x16xf32>>
    return %1 : !secret.secret<tensor<1x16xf32>>
  }
}
