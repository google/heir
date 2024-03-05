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

    let pt_1: bool = flags.input1 == 1u8;
    let pt_2: bool = flags.input2 == 1u8;

    let ct_0 = client_key.encrypt(false);
    let ct_1 = client_key.encrypt(true);


    let mut v1 = Vec::new();
    let mut v2 = Vec::new();

    for i in 0..8{
      let ct_0 = client_key.encrypt(false);
      v1.push(ct_0);
      let ct_0 = client_key.encrypt(false);
      v2.push(ct_0);
    }

    let ct_1 = client_key.encrypt(true);
    v1[1] = ct_1;
    let ct_1 = client_key.encrypt(true);
    v1[2] = ct_1.clone();

    let ct_1 = client_key.encrypt(true);
    v2[2] = ct_1.clone();
    let ct_1 = client_key.encrypt(true);
    v2[3] = ct_1.clone();

    let result = fn_under_test::fn_under_test(&server_key, &v1.to_vec(), &v2.to_vec());


    for i in 0..8{
        let output = client_key.decrypt(&result[i]);
        print!("{:?} ", output as u8);
    }


    // let output: bool = client_key.decrypt(&result);
    // print!("{:?} ", pt_1 as u8);
    // print!("{:?} ", pt_2 as u8);
    // print!("{:?} ", output as u8);
}
