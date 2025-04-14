use clap::Parser;
#[allow(unused_imports)]
use tfhe::shortint::parameters::get_parameters_from_message_and_carry;
use tfhe::shortint::*;

mod fn_under_test;

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
    let parameters = get_parameters_from_message_and_carry((1 << 3) - 1, 2);
    let (client_key, server_key) = tfhe::shortint::gen_keys(parameters);

    // query the first 2 and last 2 elements of the table
    let query = [0, 1, 254, 255];
    let ct_query: Vec<[Ciphertext; 8]> =
        query.into_iter().map(|v| encrypt_u8(v, &client_key)).collect();

    for i in 0..4 {
        let result = fn_under_test::sub_bytes(&server_key, &ct_query[i]);
        let output = decrypt_u8(&result, &client_key);
        print!("{:2X}", output);
    }
}
