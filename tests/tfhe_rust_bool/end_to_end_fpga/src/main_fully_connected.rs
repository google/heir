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
    let result = fn_under_test::fn_under_test(&server_key, &ct_1);
    let output = decrypt(&result, &client_key);

    println!("{:08b}", output);
}
