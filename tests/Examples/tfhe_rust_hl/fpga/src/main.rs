use std::time::Instant;

use clap::Parser;
use tfhe::prelude::*;
use tfhe::{generate_keys, set_server_key, BelfortServerKey, ConfigBuilder, FheUint32};

mod fn_under_test;

#[derive(Parser, Debug)]
struct Args {
    /// arguments to forward to function under test
    #[arg(id = "input_1", index = 1)]
    input1: u8,
}

// Input = 27, output = 33 * 27 + 429 = 1320
fn main() {
    let flags = Args::parse();

    let config = ConfigBuilder::default().build();

    // Client-side
    let (client_key, server_key) = generate_keys(config);

    let a: tfhe::FheUint<tfhe::FheUint32Id> = FheUint32::encrypt(flags.input1, &client_key);
    let input_vec = core::array::from_fn(|_3| core::array::from_fn(|_2| a.clone()));

    // Connect to the FPGA
    let mut fpga_key = BelfortServerKey::from(&server_key);
    fpga_key.connect();
    set_server_key(fpga_key.clone());

    let t = Instant::now();
    let result = fn_under_test::fn_under_test(&input_vec);
    let elapsed = t.elapsed();

    println!("Time elapsed: {:?}", elapsed.as_secs_f32());

    let output: u32 = result[0][0].decrypt(&client_key);
    println!("{:?}", output);
}
