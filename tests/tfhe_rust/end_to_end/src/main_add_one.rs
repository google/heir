use clap::Parser;
use tfhe::shortint::parameters::get_parameters_from_message_and_carry;
use tfhe::shortint::*;

mod fn_under_test;

// TODO(#235): improve generality
#[derive(Parser, Debug)]
struct Args {
    #[arg(id = "message_bits", long)]
    message_bits: usize,

    #[arg(id = "carry_bits", long, default_value = "2")]
    carry_bits: usize,

    /// arguments to forward to function under test
    #[arg(id = "input_1", index = 1)]
    input1: u8,
}

// Encrypt a u8
pub fn encrypt(value: u8, client_key: &ClientKey) -> Vec<Ciphertext> {
    (0..8)
        .map(|shift| {
            let bit = (value >> shift) & 1;
            client_key.encrypt(if bit != 0 { 1 } else { 0 })
        })
        .collect()
}

// Decrypt a u8
pub fn decrypt(ciphertexts: &[Ciphertext], client_key: &ClientKey) -> u8 {
    let mut accum = 0u8;
    for (i, ct) in ciphertexts.iter().enumerate() {
        let bit = client_key.decrypt(ct);
        // TODO(403): Fix the ordering of the output bits in Yosys Optimizer
        accum |= (bit as u8) << (7 - i);
    }
    accum
}

fn main() {
    let flags = Args::parse();
    let parameters =
        get_parameters_from_message_and_carry((1 << flags.message_bits) - 1, flags.carry_bits);
    let (client_key, server_key) = tfhe::shortint::gen_keys(parameters);

    let ct_1 = encrypt(flags.input1.into(), &client_key);

    let result = fn_under_test::fn_under_test(&server_key, &ct_1);
    let output = decrypt(&result, &client_key);

    println!("{:08b}", output);
}
