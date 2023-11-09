use clap::Parser;
use tfhe::shortint::parameters::get_parameters_from_message_and_carry;

mod fn_under_test;

// TODO(https://github.com/google/heir/issues/235): improve generality
#[derive(Parser, Debug)]
struct Args {
    #[arg(id = "message_bits", long)]
    message_bits: usize,

    #[arg(id = "carry_bits", long, default_value = "2")]
    carry_bits: usize,

    /// arguments to forward to function under test
    #[arg(id = "input_1", index = 1)]
    input1: u8,

    #[arg(id = "input_2", index = 2)]
    input2: u8,
}

fn main() {
    let flags = Args::parse();
    let parameters = get_parameters_from_message_and_carry((1 << flags.message_bits) - 1, flags.carry_bits);
    let (client_key, server_key) = tfhe::shortint::gen_keys(parameters);

    let ct_1 = client_key.encrypt(flags.input1.into());
    let ct_2 = client_key.encrypt(flags.input2.into());

    let result = fn_under_test::fn_under_test(&server_key, &ct_1, &ct_2);
    let output = client_key.decrypt(&result);
    println!("{:?}", output);
}
