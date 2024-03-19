use clap::Parser;
use tfhe::boolean::prelude::*;

use tfhe::boolean::engine::BooleanEngine;
use tfhe::boolean::prelude::*;
use std::time::Instant;

#[cfg(feature = "fpga")]
use tfhe::boolean::server_key::FpgaGates;


mod fn_under_test;

// TODO(https://github.com/google/heir/issues/235): improve generality
#[derive(Parser, Debug)]
struct Args {
    /// arguments to forward to function under test
    #[arg(id = "input_1", index = 1, action)]
    input1: u8,

    #[arg(id = "input_2", index = 2, action)]
    input2: u8,
}

// Encrypt a u8
pub fn encrypt(value: u8, client_key: &ClientKey) -> Vec<Ciphertext> {
    let arr: [u8; 8] = core::array::from_fn(|shift| (value >> shift) & 1 );

    let res: Vec<Ciphertext> = arr.iter()
    .map(|bit| client_key.encrypt(if *bit != 0u8 { true } else { false }))
    .collect();
    res
}

// Decrypt a u8
pub fn decrypt(ciphertexts: &Vec<Ciphertext>, client_key: &ClientKey) -> u8 {
    let mut accum = 0u8;
    for (i, ct) in ciphertexts.iter().enumerate() {
        let bit = client_key.decrypt(ct);
        accum |= (bit as u8) << i;
    }
    accum

}

fn main() {
    let flags = Args::parse();

    let params;
    let client_key;

    let mut boolean_engine = BooleanEngine::new();

    #[cfg(feature = "fpga")]
    {
      params = tfhe::boolean::engine::fpga::parameters::DEFAULT_PARAMETERS_KS_PBS;
      client_key = boolean_engine.create_client_key(*params);
    }

    #[cfg(not(feature = "fpga"))]
    {
      params = tfhe::boolean::parameters::DEFAULT_PARAMETERS_KS_PBS;
      client_key = boolean_engine.create_client_key(params);
    }

    // generate the server key, only the SW needs this
    let server_key = boolean_engine.create_server_key(&client_key);

    #[cfg(feature = "fpga")]
    server_key.enable_fpga(params);

    let ct_1 = encrypt(flags.input1.into(), &client_key);
    let ct_2 = encrypt(flags.input2.into(), &client_key);

    // let ct_1= ct_1.into_iter().collect();
    // let ct_2= ct_2.into_iter().collect();

    let result = fn_under_test::fn_under_test(&server_key, &ct_1, &ct_2);

    let output = decrypt(&result, &client_key);

    println!("{:08b}", output);
}
