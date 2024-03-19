use clap::Parser;
use tfhe::boolean::prelude::*;

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
    accum.reverse_bits()
}

fn main() {
    let flags = Args::parse();
    let (client_key, server_key) = tfhe::boolean::gen_keys();

    let ct_1 = encrypt(flags.input1.into(), &client_key);
    let ct_2 = encrypt(flags.input2.into(), &client_key);


    let result = fn_under_test::fn_under_test(&server_key, &ct_1, &ct_2);

    let output = decrypt(&result, &client_key);

    println!("{:08b}", output);
}
