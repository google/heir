use tfhe::prelude::*;
use tfhe::{ClientKey, FheUint8};

type Ciphertext = tfhe::FheUint<tfhe::FheUint8Id>;

pub fn encrypt_block(value: [u8; 16], client_key: &ClientKey) -> [Ciphertext; 16] {
    let c_vec: Vec<Ciphertext> =
        value.into_iter().map(|v| FheUint8::encrypt(v, client_key)).collect();
    let c_arr: [Ciphertext; 16] = c_vec.try_into().unwrap_or_else(|_| panic!("failed"));
    c_arr
}

pub fn decrypt_block(ciphertexts: &[Ciphertext; 16], client_key: &ClientKey) -> Vec<u8> {
    ciphertexts.into_iter().map(|ct| ct.decrypt(&client_key)).collect()
}

#[cfg(test)]
mod test {
    use add_round_key_test_rs_lib;

    use super::*;
    use tfhe::{generate_keys, set_server_key, ConfigBuilder};
    // use std::time::Instant;

    #[test]
    fn simple_test() {
        // Test data
        let block = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let key = [15, 21, 113, 201, 71, 217, 232, 89, 12, 183, 173, 214, 175, 127, 103, 152];
        let expected = [14, 23, 114, 205, 66, 223, 239, 81, 5, 189, 166, 218, 162, 113, 104, 136];

        // Create Keys
        let config = ConfigBuilder::default().build();
        let (client_key, server_key) = generate_keys(config);
        set_server_key(server_key);

        // Encrypt Values
        let encrypted_block = encrypt_block(block, &client_key);
        let encrypted_key = encrypt_block(key, &client_key);

        // Encrypted Calculations
        // let time_start = Instant::now();
        let result = add_round_key_test_rs_lib::add_round_key(&encrypted_block, &encrypted_key);
        // let time_end = time_start.elapsed();
        let output = decrypt_block(&result, &client_key);
        assert_eq!(output, expected);
        // println!("Success");
        // println!("Execution time {:?}", time_end);
    }
}
