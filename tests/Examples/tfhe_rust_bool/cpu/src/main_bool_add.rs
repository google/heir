use tfhe::boolean::prelude::*;

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

#[cfg(test)]
mod test {
    use super::encrypt;
    use super::decrypt;

    use bool_add_test_rs_lib;

    #[test]
    fn simple_test() {
        let (client_key, server_key) = tfhe::boolean::gen_keys();

        let ct_1 = encrypt(15, &client_key);
        let ct_2 = encrypt(3, &client_key);

        let result = bool_add_test_rs_lib::fn_under_test(&server_key, &ct_1, &ct_2);

        let output = decrypt(&result, &client_key);

        assert_eq!(output, 18);
    }
}
