#ifndef LIB_TARGET_TFHERUST_TFHERUSTTEMPLATES_H_
#define LIB_TARGET_TFHERUST_TFHERUSTTEMPLATES_H_

#include <string_view>

namespace mlir {
namespace heir {
namespace tfhe_rust {

constexpr std::string_view kModulePrelude = R"rust(
use tfhe::shortint::Ciphertext;
use tfhe::shortint::ServerKey;
use tfhe::shortint::server_key::LookupTableOwned;

use std::collections::HashMap;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

enum GateInput {
    Tv(usize), // key in a global hashmap
}

use GateInput::*;

enum OpType<'a> {
    LUT3(&'a str), // key in a global hashmap
    ADD,
    LSH(u8), // shift value
}

use OpType::*;
)rust";

constexpr std::string_view kRunLevelDefn = R"rust(
let lut3 = |args: &[&Ciphertext], lut: &LookupTableOwned, server_key: &ServerKey| -> Ciphertext {
    return server_key.apply_lookup_table(args[0], lut);
};

let add = |args: &[&Ciphertext], server_key: &ServerKey| -> Ciphertext {
    return server_key.unchecked_add(args[0], args[1]);
};

let left_shift = |args: &[&Ciphertext], shift: u8, server_key: &ServerKey| -> Ciphertext {
    return server_key.scalar_left_shift(args[0], shift);
};

let run_level = |
  server_key: &ServerKey,
  temp_nodes: &mut HashMap<usize, Ciphertext>,
  luts: &mut HashMap<&str, LookupTableOwned>,
  tasks: &[((OpType, usize), &[GateInput])]
| {
    let updates = tasks
        .into_par_iter()
        .map(|(k, task_args)| {
            let (op_type, result) = k;
            let task_args = task_args.into_iter()
              .map(|arg| match arg {
                Tv(ndx) => &temp_nodes[&ndx],
              }).collect::<Vec<_>>();
            let op = |args: &[&Ciphertext]| match op_type {
              LUT3(lut) => lut3(args, &luts[lut], server_key),
              ADD => add(args, server_key),
              LSH(shift) => left_shift(args, *shift, server_key)
            };
            ((result), op(&task_args))
        })
        .collect::<Vec<_>>();
    updates.into_iter().for_each(|(id, v)| {
      temp_nodes.insert(*id, v);
    });
};
)rust";

}  // namespace tfhe_rust
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_TFHERUST_TFHERUSTTEMPLATES_H_
