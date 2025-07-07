#[allow(unused_imports)]
use std::time::Instant;
use tfhe::shortint::*;

// Encrypt a u8
pub fn encrypt(value: u8, client_key: &ClientKey) -> [[[Ciphertext; 8]; 1]; 1] {
    core::array::from_fn(|_| {
        core::array::from_fn(|_| {
            core::array::from_fn(|shift| {
                let bit = (value >> shift) & 1;
                client_key.encrypt(if bit != 0 { 1 } else { 0 })
            })
        })
    })
}

// Decrypt a u8
pub fn decrypt(ciphertexts: &[Ciphertext], client_key: &ClientKey) -> u8 {
    let mut accum = 0u8;
    for (i, ct) in ciphertexts.iter().enumerate() {
        let bit = client_key.decrypt(ct);
        accum |= (bit as u8) << i;
    }
    accum
}

#[cfg(test)]
mod test {
    use tfhe::shortint::parameters::get_parameters_from_message_and_carry;

    use super::encrypt;
    use super::decrypt;

    use fully_connected_test_rs_lib;

    #[test]
    fn simple_test() {
        let parameters = get_parameters_from_message_and_carry((1 << 3) - 1, 2);
        let (client_key, server_key) = tfhe::shortint::gen_keys(parameters);

        let ct_1 = encrypt(2, &client_key);

        let result = fully_connected_test_rs_lib::fn_under_test(&server_key, &ct_1);
        let output = decrypt(&result[0][0][0..8], &client_key);

        assert_eq!(output, 5);
    }
}
