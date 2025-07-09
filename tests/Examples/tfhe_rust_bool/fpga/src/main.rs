#![allow(dead_code)]
use clap::Parser;

use server_key_enum::ServerKeyEnum;
use tfhe::core_crypto::commons::generators::DeterministicSeeder;
use tfhe::core_crypto::commons::math::random::Seed;
use tfhe::core_crypto::prelude::ActivatedRandomGenerator;

use std::time::Instant;
use tfhe::boolean::engine::fpga::{BelfortBooleanServerKey, Gate};
use tfhe::boolean::engine::BooleanEngine;
use tfhe::boolean::prelude::*;

mod fn_under_test;
mod server_key_enum;

const FPGA_COUNT: usize = 1;
const SEED: u64 = 0;

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
    accum.reverse_bits()
}

fn main() {
    // Deterministic engine
    let mut seeder = DeterministicSeeder::<ActivatedRandomGenerator>::new(Seed(SEED as u128));
    let boolean_engine = BooleanEngine::new_from_seeder(&mut seeder);
    BooleanEngine::replace_thread_local(boolean_engine);

    let params = tfhe::boolean::parameters::DEFAULT_PARAMETERS_KS_PBS;

    let client_key: ClientKey = ClientKey::new(&params);
    let server_key: ServerKey = ServerKey::new(&client_key);

    let flags = Args::parse();

    let key_wrapped;

    #[cfg(feature = "fpga")]
    {
        let mut fpga_key: BelfortBooleanServerKey =
            BelfortBooleanServerKey::from(server_key.clone());
        fpga_key.connect(FPGA_COUNT);
        key_wrapped = ServerKeyEnum::TypeFPGA(fpga_key);
    }

    #[cfg(not(feature = "fpga"))]
    {
        key_wrapped = ServerKeyEnum::TypeSW(server_key.clone());
    }

    let ct_1 = encrypt(flags.input1.into(), &client_key);
    let ct_2 = encrypt(flags.input2.into(), &client_key);

    // timing placeholders to quickly obtain the measurements of the generated function
    let t = Instant::now();

    let result = fn_under_test::fn_under_test(&key_wrapped, &ct_1, &ct_2);

    let run = t.elapsed().as_secs_f64();
    println!("{:.3?} ms", run * 1000.0);

    let output = decrypt(&result, &client_key);

    println!("{:08b}", output);
}
