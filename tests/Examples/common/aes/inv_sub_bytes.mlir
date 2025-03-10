// Inverse Byte substitution function.

#map = affine_map<(d0) -> (d0)>
module {
  func.func @inv_sub_bytes(%arg0: tensor<16xi8> {secret.secret}) -> tensor<16xi8> {
    %cst = arith.constant dense<"0x52096ad53036a538bf40a39e81f3d7fb7ce339829b2fff87348e4344c4dee9cb547b9432a6c2233dee4c950b42fac34e082ea16628d924b2765ba2496d8bd12572f8f66486689816d4a45ccc5d65b6926c704850fdedb9da5e154657a78d9d8490d8ab008cbcd30af7e45805b8b34506d02c1e8fca3f0f02c1afbd0301138a6b3a9111414f67dcea97f2cfcef0b4e67396ac7422e7ad3585e2f937e81c75df6e47f11a711d29c5896fb7620eaa18be1bfc563e4bc6d279209adbc0fe78cd5af41fdda8338807c731b11210592780ec5f60517fa919b54a0d2de57a9f93c99cefa0e03b4dae2af5b0c8ebbb3c83539961172b047eba77d626e169146355210c7d"> : tensor<256xi8>
    %0 = tensor.empty() : tensor<16xi8>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : tensor<16xi8>) outs(%0 : tensor<16xi8>) {
    ^bb0(%in: i8, %out: i8):
      %2 = arith.index_cast %in : i8 to index
      %extracted = tensor.extract %cst[%2] : tensor<256xi8>
      linalg.yield %extracted : i8
    } -> tensor<16xi8>
    return %1 : tensor<16xi8>
  }
}
