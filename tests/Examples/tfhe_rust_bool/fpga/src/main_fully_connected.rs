#[allow(unused_imports)]
use std::time::Instant;

use clap::Parser;

use tfhe::boolean::engine::BooleanEngine;
use tfhe::boolean::prelude::*;
#[cfg(feature = "fpga")]
use tfhe::boolean::server_key::FpgaAcceleration;

mod fn_under_test;

#[derive(Parser, Debug)]
struct Args {
    /// arguments to forward to function under test
    #[arg(id = "input_1", index = 1)]
    input1: u8,
}

// Encrypt a u8
pub fn encrypt(value: u8, client_key: &ClientKey) -> Vec<Ciphertext> {
    let arr: [u8; 8] = core::array::from_fn(|shift| (value >> shift) & 1);

    let res: Vec<Ciphertext> =
        arr.iter().map(|bit| client_key.encrypt(if *bit != 0u8 { true } else { false })).collect();
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
        params = tfhe::core_crypto::fpga::parameters::DEFAULT_PARAMETERS_KS_PBS;
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
    server_key.enable_fpga(params, 1);

    let ct_1 = encrypt(flags.input1.into(), &client_key);

    // timing placeholders to quickly obtain the measurements of the generated function
    let t = Instant::now();

    let result = fn_under_test::fn_under_test(&server_key, &ct_1);

    let run = t.elapsed().as_millis();
    println!("{:?}", run);

    let output = decrypt(&result, &client_key);

    println!("{:08b}", output);
}
