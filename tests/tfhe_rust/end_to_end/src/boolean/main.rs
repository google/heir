use clap::Parser;

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

fn main() {
    let flags = Args::parse();
    let (client_key, server_key) = tfhe::boolean::gen_keys();

    let ct_1 = client_key.encrypt(flags.input1 == 0u8);
    let ct_2 = client_key.encrypt(flags.input2 == 0u8);

    let result = fn_under_test::fn_under_test(&server_key, &ct_1, &ct_2);
    let output: bool = client_key.decrypt(&result);
    println!("{:?}", output as u8);
}
