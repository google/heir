use clap::Parser;
#[allow(unused_imports)]
use tfhe::shortint::parameters::get_parameters_from_message_and_carry;
use tfhe::shortint::*;

mod fn_under_test;

#[derive(Parser, Debug)]
struct Args {
    /// arguments to forward to function under test
    #[arg(id = "input_1", index = 1)]
    input1: u8,
}

pub fn encrypt_u8(value: u8, client_key: &ClientKey) -> [Ciphertext; 8] {
    core::array::from_fn(|shift| {
        let bit = (value >> shift) & 1;
        client_key.encrypt(if bit != 0 { 1 } else { 0 })
    })
}

pub fn decrypt_u8(ciphertexts: &[Ciphertext; 8], client_key: &ClientKey) -> u8 {
    let mut accum = 0u8;
    for (i, ct) in ciphertexts.iter().enumerate() {
        let bit = client_key.decrypt(ct);
        accum |= (bit as u8) << i;
    }
    accum
}

fn main() {
    let flags = Args::parse();

    let parameters = get_parameters_from_message_and_carry((1 << 3) - 1, 2);
    let (client_key, server_key) = tfhe::shortint::gen_keys(parameters);

    let ct_1 = encrypt_u8(flags.input1.into(), &client_key);

    let (result1, result2) = fn_under_test::multi_output(&server_key, &ct_1);
    let output1 = decrypt_u8(&result1, &client_key);
    let output2 = decrypt_u8(&result2, &client_key);
    println!("{:08b} {:08b}", output1, output2);
}
